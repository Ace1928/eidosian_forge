import multiprocessing.dummy
import queue
import random
import threading
import time
import warnings
import weakref
from contextlib import closing
import numpy as np
import tree
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def _set_tf_output_signature(self):
    from keras.src.utils.module_utils import tensorflow as tf

    def get_tensor_spec(x):
        shape = x.shape
        if len(shape) < 1:
            raise ValueError(f'The arrays returned by PyDataset.__getitem__() must be at least rank 1. Received: {x} of rank {len(x.shape)}')
        shape = list(shape)
        shape[0] = None
        dtype = backend.standardize_dtype(x.dtype)
        return tf.TensorSpec(shape=shape, dtype=dtype)
    batch = self.py_dataset[0]
    batch = self._standardize_batch(batch)
    self._output_signature = tree.map_structure(get_tensor_spec, batch)