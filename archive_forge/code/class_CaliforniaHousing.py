import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import autokeras as ak
from benchmark.experiments import experiment
class CaliforniaHousing(StructuredDataRegressorExperiment):

    @staticmethod
    def load_data():
        house_dataset = sklearn.datasets.fetch_california_housing()
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(house_dataset.data, np.array(house_dataset.target), test_size=0.2, random_state=42)
        return ((x_train, y_train), (x_test, y_test))