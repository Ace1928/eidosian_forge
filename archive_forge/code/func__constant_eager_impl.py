import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
def _constant_eager_impl(ctx, value, dtype, shape, verify_shape):
    """Creates a constant on the current device."""
    t = convert_to_eager_tensor(value, ctx, dtype)
    if shape is None:
        return t
    shape = tensor_shape.as_shape(shape)
    if shape == t.shape:
        return t
    if verify_shape:
        raise TypeError(f'Expected Tensor {t} (converted from {value}) with shape {tuple(shape)}, but got shape {tuple(t.shape)}.')
    num_t = t.shape.num_elements()
    if num_t == shape.num_elements():
        return _eager_reshape(t, shape.as_list(), ctx)
    if num_t == 1:
        if t.dtype == dtypes.bool:
            with ops.device('/device:CPU:0'):
                x = _eager_fill(shape.as_list(), _eager_identity(t, ctx), ctx)
            return _eager_identity(x, ctx)
        else:
            return _eager_fill(shape.as_list(), t, ctx)
    raise TypeError(f'Eager execution of tf.constant with unsupported shape. Tensor {t} (converted from {value}) has {num_t:d} elements, but got `shape` {shape} with {shape.num_elements()} elements).')