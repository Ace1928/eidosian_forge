import builtins
import numbers
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _OverrideBinaryOperatorHelper(func, op_name, clazz_object=tensor_lib.Tensor):
    """Register operators with different tensor and scalar versions.

  If `clazz_object` is `SparseTensor`, assumes `func` takes `(sp_indices,
  sp_values, sp_shape, dense)` and outputs `(new_sp_values)`.

  Args:
    func: the operator
    op_name: name of the operator being overridden
    clazz_object: class to override for.  Either `Tensor` or `SparseTensor`.
  """

    @traceback_utils.filter_traceback
    def binary_op_wrapper(x, y):
        with ops.name_scope(None, op_name, [x, y]) as name:
            try:
                x, y = maybe_promote_tensors(x, y)
                return func(x, y, name=name)
            except (TypeError, ValueError) as e:
                if hasattr(type(y), '__r%s__' % op_name):
                    try:
                        r_op = getattr(y, '__r%s__' % op_name)
                        out = r_op(x)
                        if out is NotImplemented:
                            raise
                        return out
                    except (TypeError, ValueError):
                        raise e
                else:
                    raise

    @traceback_utils.filter_traceback
    def binary_op_wrapper_sparse(sp_x, y):
        with ops.name_scope(None, op_name, [sp_x, y]) as name:
            y = ops.convert_to_tensor(y, dtype=sp_x.dtype.base_dtype, name='y')
            return sparse_tensor.SparseTensor(sp_x.indices, func(sp_x.indices, sp_x.values, sp_x.dense_shape, y, name=name), sp_x.dense_shape)

    @traceback_utils.filter_traceback
    def r_binary_op_wrapper(y, x):
        with ops.name_scope(None, op_name, [x, y]) as name:
            y, x = maybe_promote_tensors(y, x, force_same_dtype=True)
            return func(x, y, name=name)
    try:
        doc = func.__doc__
    except AttributeError:
        doc = None
    binary_op_wrapper.__doc__ = doc
    r_binary_op_wrapper.__doc__ = doc
    binary_op_wrapper_sparse.__doc__ = doc
    if clazz_object is tensor_lib.Tensor:
        clazz_object._override_operator('__%s__' % op_name, binary_op_wrapper)
        del binary_op_wrapper
        clazz_object._override_operator('__r%s__' % op_name, r_binary_op_wrapper)
        del r_binary_op_wrapper
    else:
        clazz_object._override_operator('__%s__' % op_name, binary_op_wrapper_sparse)
        del binary_op_wrapper_sparse