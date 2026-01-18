import numpy as np
import pandas as pd
import tensorflow as tf
from autokeras.engine import adapter as adapter_module
class StructuredDataAdapter(adapter_module.Adapter):

    def check(self, x):
        if not isinstance(x, (pd.DataFrame, np.ndarray, tf.data.Dataset)):
            raise TypeError('Unsupported type {type} for {name}.'.format(type=type(x), name=self.__class__.__name__))

    def convert_to_dataset(self, dataset, batch_size):
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        if isinstance(dataset, np.ndarray) and dataset.dtype == np.object:
            dataset = dataset.astype(np.unicode)
        return super().convert_to_dataset(dataset, batch_size)