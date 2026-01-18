import hashlib
import numbers
import sys
import types as python_types
import warnings
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
def _variational_recurrent_dropout_value(self, unused_index, value, noise, keep_prob):
    """Performs dropout given the pre-calculated noise tensor."""
    random_tensor = keep_prob + noise
    binary_tensor = math_ops.floor(random_tensor)
    ret = math_ops.divide(value, keep_prob) * binary_tensor
    ret.set_shape(value.get_shape())
    return ret