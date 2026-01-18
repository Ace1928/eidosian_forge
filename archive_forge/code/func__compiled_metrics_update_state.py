import contextlib
import warnings
import numpy as np
import tensorflow as tf
import tree
from packaging.version import Version
from tensorflow.python.eager import context as tf_context
from keras.src import callbacks as callbacks_module
from keras.src import metrics as metrics_module
from keras.src import optimizers as optimizers_module
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
def _compiled_metrics_update_state(self, y, y_pred, sample_weight=None):
    warnings.warn('`model.compiled_metrics()` is deprecated. Instead, use e.g.:\n```\nfor metric in self.metrics:\n    metric.update_state(y, y_pred)\n```\n', stacklevel=2)
    for metric in self.metrics:
        if isinstance(metric, metrics_module.Mean):
            metric.update_state(y_pred, sample_weight=sample_weight)
        else:
            metric.update_state(y, y_pred, sample_weight=sample_weight)