import collections
import multiprocessing
import os
import threading
import warnings
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset
from keras.src.utils import image_utils
from keras.src.utils import io_utils
from keras.src.utils.module_utils import scipy
def _flow_index(self):
    self.reset()
    while 1:
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        if self.batch_index == 0:
            self._set_index_array()
        if self.n == 0:
            current_index = 0
        else:
            current_index = self.batch_index * self.batch_size % self.n
        if self.n > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0
        self.total_batches_seen += 1
        yield self.index_array[current_index:current_index + self.batch_size]