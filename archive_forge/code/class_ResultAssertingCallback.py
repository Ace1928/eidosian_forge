import os
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
import keras.src as keras
from keras.src import callbacks as callbacks_lib
from keras.src.engine import sequential
from keras.src.layers import core as core_layers
from keras.src.layers.preprocessing import string_lookup
from keras.src.optimizers.legacy import gradient_descent
from keras.src.utils import dataset_creator
from tensorflow.python.platform import tf_logging as logging
class ResultAssertingCallback(callbacks_lib.Callback):
    """A callback that asserts the result of the tests."""

    def __init__(self):
        self._prev_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        logging.info('testModelFit: epoch=%r, logs=%r', epoch, logs)
        if epoch <= self._prev_epoch:
            raise RuntimeError('Epoch is supposed to be larger than previous.')
        self._prev_epoch = epoch
        is_loss_float = logs.get('loss', None) is not None and isinstance(logs['loss'], (float, np.floating))
        if not is_loss_float:
            raise RuntimeError('loss is supposed to be in the logs and float.')