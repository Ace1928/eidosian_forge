import re
import numpy as np
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import array_ops_stack as _array_ops_stack
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _tile_for_broadcasting(matrix, t):
    expanded = _array_ops.reshape(matrix, _array_ops.concat([_array_ops.ones([_array_ops.rank(t) - 2], _dtypes.int32), _array_ops.shape(matrix)], 0))
    return _array_ops.tile(expanded, _array_ops.concat([_array_ops.shape(t)[:-2], [1, 1]], 0))