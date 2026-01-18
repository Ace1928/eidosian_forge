from tensorflow.core.config import flags
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
def FlattenNestedIndexedSlices(grad):
    assert isinstance(grad, indexed_slices.IndexedSlices)
    if isinstance(grad.values, tensor_lib.Tensor):
        return grad
    else:
        assert isinstance(grad.values, indexed_slices.IndexedSlices)
        g = FlattenNestedIndexedSlices(grad.values)
        return indexed_slices.IndexedSlices(g.values, array_ops.gather(grad.indices, g.indices), g.dense_shape)