# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement 

This experiment aims to build an LSTM-based neural network model for named entity recognition utilizing bidirectional LSTM-based model. The dataset contains numerous sentences, each with numerous words and their corresponding tags. To train our model, we vectorize these phrases utilizing Embedding layer method.Recurrent neural networks that operate in both directions (bi-directional) and can link two or more hidden layers to a single output. The output layer can simultaneously receive input from past and future states to predict the next word.

## Dataset

![Screenshot 2024-10-04 203959](https://github.com/user-attachments/assets/869231a8-e5ae-4888-9827-4eb706a08984)



## DESIGN STEPS

## STEP 1:
Import the necessary packages.

## STEP 2:
Read the dataset, and fill the null values using forward fill.

## STEP 3:
Create a list of words, and tags. Also find the number of unique words and tags in the dataset.

## Step 4 :
Create a dictionary for the words and their Index values. Do the same for the tags as well,Now we move to moulding the data for training and testing.

## Step 5 :
We do this by padding the sequences,This is done to acheive the same length of input data.

## Step 6 :
We build a build a model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, Time Distributed Dense Layers.

## Step 7 :
We compile the model and fit the train sets and validation sets,We plot the necessary graphs for analysis,A custom prediction is done to test the model manually.

## PROGRAM
### Name:SRIVATSAN G
### Register Number:212223230216
````
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model
data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data.head(50)
data = data.fillna(method="ffill")
data.head(50)
print("Unique tags are:", tags)
num_words = len(words)
num_tags = len(tags)
class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
getter = SentenceGetter(data)
sentences = getter.sentences
len(sentences)
sentences[0]
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
word2idx
plt.hist([len(s) for s in sentences], bins=50)
plt.show()
X1 = [[word2idx[w[0]] for w in s] for s in sentences]
type(X1[0])
X1[0]
max_len = 50
X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)
X[0]
y1 = [[tag2idx[w[2]] for w in s] for s in sentences]
y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=1)
X_train.shape
X_train[0]
y_train[0]

input_word = layers.Input(shape=(max_len,))
embedding_layer = layers.Embedding(input_dim=num_words,
                                  output_dim=50,
                                  input_length=max_len)(input_word)
dropout_layer = layers.SpatialDropout1D(0.1)(embedding_layer)
bidirectional_lstm = layers.Bidirectional(
    layers.LSTM(units=100, return_sequences=True,
                recurrent_dropout=0.1))(dropout_layer)
output = layers.TimeDistributed(
    layers.Dense(num_tags, activation="softmax"))(bidirectional_lstm)
model = Model(input_word,output)
print('Name: SRIVATSAN G   Register Number:212223230216')
model.summary()
model.compile(optimizer='adam',  # Or any other optimizer
              loss='sparse_categorical_crossentropy',  # Or any other appropriate loss function
              metrics=['accuracy'])  # Or any other metrics you want to monitor

history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test,y_test),
    batch_size=32,
    epochs=3,
)
metrics = pd.DataFrame(model.history.history)
metrics.head()
     
print('Name: Srivatsan G             Register Number:212223230216')
metrics[['accuracy','val_accuracy']].plot()
     
print('Name:SRIVATSAN G              Register Number:212223230216       ')
metrics[['loss','val_loss']].plot()
i = 20
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print('Name:SRIVATSAN G                 Register Number:212223230216')
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))

````
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2024-10-04 204316](https://github.com/user-attachments/assets/2b7258fc-e246-4b9c-8843-312c8d0a300f)

![Screenshot 2024-10-04 204328](https://github.com/user-attachments/assets/3fd39451-e348-463f-9fbc-21d921540628)



### Sample Text Prediction
![Screenshot 2024-10-04 204551](https://github.com/user-attachments/assets/1b8e391c-2440-46ee-8ceb-90e03506442a)

![Screenshot 2024-10-04 204622](https://github.com/user-attachments/assets/99a5bf24-119d-4624-8022-d5f02a1090e3)

![Screenshot 2024-10-04 204650](https://github.com/user-attachments/assets/022317ce-22cf-45e5-a600-5572f329571c)

![Screenshot 2024-10-04 204736](https://github.com/user-attachments/assets/0d7e42b1-18b9-4874-9bf8-aa3362ed5233)

## RESULT

Thus, an LSTM-based model (bi-directional) for recognizing the named entities in the text is developed Successfully.


