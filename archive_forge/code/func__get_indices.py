import builtins
import collections
import math
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tfnp
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.tensorflow import sparse
from keras.src.backend.tensorflow.core import convert_to_tensor
def _get_indices(method):
    """Get values of y at the indices implied by method."""
    if method == 'lower':
        indices = tf.math.floor((d - 1) * q)
    elif method == 'higher':
        indices = tf.math.ceil((d - 1) * q)
    elif method == 'nearest':
        indices = tf.round((d - 1) * q)
    return tf.clip_by_value(tf.cast(indices, 'int32'), 0, tf.shape(y)[-1] - 1)