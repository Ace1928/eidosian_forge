import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.gen_sparse_ops import *
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import tf_export
class _UnaryMapValueDispatcher(dispatch.OpDispatcher):
    """OpDispatcher for unary ops that maps base function across sparse values."""

    def __init__(self, original_func):
        self._original_func = original_func
        func_name = get_canonical_name_for_symbol(original_func)
        arg_names = tf_inspect.getfullargspec(original_func)[0]
        self._x = arg_names[0]
        original_func.__doc__ = original_func.__doc__.rstrip() + '\n\n' + '    If `{x}` is a `SparseTensor`, returns\n    `SparseTensor({x}.indices, tf.{func}({x}.values, ...), {x}.dense_shape)`'.format(x=self._x, func=func_name)

    def handle(self, args, kwargs):
        if args:
            x, args = (args[0], args[1:])
        else:
            kwargs = kwargs.copy()
            x = kwargs.pop(self._x, None)
        if isinstance(x, sparse_tensor.SparseTensor):
            return sparse_tensor.SparseTensor(indices=x.indices, values=self._original_func(x.values, *args, **kwargs), dense_shape=x.dense_shape)
        else:
            return self.NOT_SUPPORTED