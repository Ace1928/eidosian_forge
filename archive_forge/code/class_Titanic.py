import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import autokeras as ak
from benchmark.experiments import experiment
class Titanic(StructuredDataClassifierExperiment):

    def __init__(self):
        super().__init__(name='Titanic')

    @staticmethod
    def load_data():
        TRAIN_DATA_URL = 'https://storage.googleapis.com/tf-datasets/titanic/train.csv'
        TEST_DATA_URL = 'https://storage.googleapis.com/tf-datasets/titanic/eval.csv'
        x_train = tf.keras.utils.get_file('titanic_train.csv', TRAIN_DATA_URL)
        x_test = tf.keras.utils.get_file('titanic_eval.csv', TEST_DATA_URL)
        return ((x_train, 'survived'), (x_test, 'survived'))