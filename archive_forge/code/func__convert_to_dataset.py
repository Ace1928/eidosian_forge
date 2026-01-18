from pathlib import Path
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import graph as graph_module
from autokeras import pipeline
from autokeras import tuners
from autokeras.engine import head as head_module
from autokeras.engine import node as node_module
from autokeras.engine import tuner
from autokeras.nodes import Input
from autokeras.utils import data_utils
from autokeras.utils import utils
def _convert_to_dataset(self, x, y, validation_data, batch_size):
    """Convert the data to tf.data.Dataset."""
    self._check_data_format((x, y))
    if isinstance(x, tf.data.Dataset):
        dataset = x
        x = dataset.map(lambda x, y: x)
        y = dataset.map(lambda x, y: y)
    x = self._adapt(x, self.inputs, batch_size)
    y = self._adapt(y, self._heads, batch_size)
    dataset = tf.data.Dataset.zip((x, y))
    if validation_data:
        self._check_data_format(validation_data, validation=True)
        if isinstance(validation_data, tf.data.Dataset):
            x = validation_data.map(lambda x, y: x)
            y = validation_data.map(lambda x, y: y)
        else:
            x, y = validation_data
        x = self._adapt(x, self.inputs, batch_size)
        y = self._adapt(y, self._heads, batch_size)
        validation_data = tf.data.Dataset.zip((x, y))
    return (dataset, validation_data)