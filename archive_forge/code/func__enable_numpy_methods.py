import numbers
import sys
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export
def _enable_numpy_methods(tensor_class):
    """A helper method for adding additional NumPy methods."""
    t = property(_tensor_t)
    setattr(tensor_class, 'T', t)
    ndim = property(_tensor_ndim)
    setattr(tensor_class, 'ndim', ndim)
    size = property(_tensor_size)
    setattr(tensor_class, 'size', size)
    setattr(tensor_class, '__pos__', _tensor_pos)
    setattr(tensor_class, 'tolist', _tensor_tolist)
    setattr(tensor_class, 'transpose', np_array_ops.transpose)
    setattr(tensor_class, 'flatten', np_array_ops.flatten)
    setattr(tensor_class, 'reshape', np_array_ops._reshape_method_wrapper)
    setattr(tensor_class, 'ravel', np_array_ops.ravel)
    setattr(tensor_class, 'clip', clip)
    setattr(tensor_class, 'astype', math_ops.cast)
    setattr(tensor_class, '__round__', np_array_ops.around)
    setattr(tensor_class, 'max', np_array_ops.amax)
    setattr(tensor_class, 'mean', np_array_ops.mean)
    setattr(tensor_class, 'min', np_array_ops.amin)
    data = property(lambda self: self)
    setattr(tensor_class, 'data', data)