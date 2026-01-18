import abc
import contextlib
import functools
import itertools
import math
import random
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.distribute import distributed_training_utils
from keras.src.engine import training_utils
from keras.src.utils import data_utils
from keras.src.utils import dataset_creator
from keras.src.utils import tf_utils
from tensorflow.python.distribute.input_lib import (
from tensorflow.python.eager import context
from tensorflow.python.framework import type_spec
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.data.ops import (
from tensorflow.python.data.ops import from_generator_op
from tensorflow.python.data.ops import range_op
from tensorflow.python.data.ops import from_tensors_op
from tensorflow.python.data.ops import from_tensor_slices_op
def _convert_single_tensor(x):
    if _is_pandas_series(x):
        x = np.expand_dims(x.to_numpy(), axis=-1)
    if isinstance(x, np.ndarray):
        dtype = None
        if issubclass(x.dtype.type, np.floating):
            dtype = backend.floatx()
        return tf.convert_to_tensor(x, dtype=dtype)
    elif _is_scipy_sparse(x):
        return _scipy_sparse_to_sparse_tensor(x)
    return x