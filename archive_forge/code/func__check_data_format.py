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
def _check_data_format(self, dataset, validation=False, predict=False):
    """Check if the dataset has the same number of IOs with the model."""
    if validation:
        in_val = ' in validation_data'
        if isinstance(dataset, tf.data.Dataset):
            x = dataset
            y = None
        else:
            x, y = dataset
    else:
        in_val = ''
        x, y = dataset
    if isinstance(x, tf.data.Dataset) and y is not None:
        raise ValueError('Expected y to be None when x is tf.data.Dataset{in_val}.'.format(in_val=in_val))
    if isinstance(x, tf.data.Dataset):
        if not predict:
            x_shapes, y_shapes = data_utils.dataset_shape(x)
            x_shapes = nest.flatten(x_shapes)
            y_shapes = nest.flatten(y_shapes)
        else:
            x_shapes = nest.flatten(data_utils.dataset_shape(x))
    else:
        x_shapes = [a.shape for a in nest.flatten(x)]
        if not predict:
            y_shapes = [a.shape for a in nest.flatten(y)]
    if len(x_shapes) != len(self.inputs):
        raise ValueError('Expected x{in_val} to have {input_num} arrays, but got {data_num}'.format(in_val=in_val, input_num=len(self.inputs), data_num=len(x_shapes)))
    if not predict and len(y_shapes) != len(self.outputs):
        raise ValueError('Expected y{in_val} to have {output_num} arrays, but got {data_num}'.format(in_val=in_val, output_num=len(self.outputs), data_num=len(y_shapes)))