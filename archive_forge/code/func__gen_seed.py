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
def _gen_seed(self, salt_prefix, index):
    if self._seed is None:
        return None
    salt = '%s_%d' % (salt_prefix, index)
    string = (str(self._seed) + salt).encode('utf-8')
    return int(hashlib.md5(string).hexdigest()[:8], 16) & 2147483647