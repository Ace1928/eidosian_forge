import os
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_files
import autokeras as ak
from benchmark.experiments import experiment
class IMDB(experiment.Experiment):

    def __init__(self):
        super().__init__(name='IMDB')

    def get_auto_model(self):
        return ak.TextClassifier(max_trials=10, directory=self.tmp_dir, overwrite=True)

    @staticmethod
    def load_data():
        dataset = tf.keras.utils.get_file(fname='aclImdb.tar.gz', origin='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', extract=True)
        IMDB_DATADIR = os.path.join(os.path.dirname(dataset), 'aclImdb')
        classes = ['pos', 'neg']
        train_data = load_files(os.path.join(IMDB_DATADIR, 'train'), shuffle=True, categories=classes)
        test_data = load_files(os.path.join(IMDB_DATADIR, 'test'), shuffle=False, categories=classes)
        x_train = np.array(train_data.data)
        y_train = np.array(train_data.target)
        x_test = np.array(test_data.data)
        y_test = np.array(test_data.target)
        return ((x_train, y_train), (x_test, y_test))