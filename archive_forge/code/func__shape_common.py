import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _shape_common(s1, s2):
    """The greatest lower bound (ordered by specificity) TensorShape."""
    s1 = tensor_shape.TensorShape(s1)
    s2 = tensor_shape.TensorShape(s2)
    if s1.ndims is None or s2.ndims is None or s1.ndims != s2.ndims:
        return tensor_shape.unknown_shape()
    d = [d1 if d1 is not None and d1 == d2 else None for d1, d2 in zip(s1.as_list(), s2.as_list())]
    return tensor_shape.TensorShape(d)