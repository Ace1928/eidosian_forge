import builtins
import enum
import functools
import math
import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
def _remove_indices(a, b):
    """Remove indices (`b`) from `a`."""
    items = array_ops_stack.unstack(sort_ops.sort(array_ops_stack.stack(b)), num=len(b))
    i = 0
    result = []
    for item in items:
        result.append(a[i:item])
        i = item + 1
    result.append(a[i:])
    return array_ops.concat(result, 0)