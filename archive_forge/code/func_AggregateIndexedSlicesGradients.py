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
def AggregateIndexedSlicesGradients(grads):
    """Aggregates gradients containing `IndexedSlices`s."""
    if len(grads) < 1:
        return None
    if len(grads) == 1:
        return grads[0]
    grads = [g for g in grads if g is not None]
    if any((isinstance(g, tensor_lib.Tensor) for g in grads)):
        return math_ops.add_n(grads)
    grads = math_ops._as_indexed_slices_list(grads)
    grads = [FlattenNestedIndexedSlices(x) for x in grads]
    concat_grad = indexed_slices.IndexedSlices(array_ops.concat([x.values for x in grads], axis=0), array_ops.concat([x.indices for x in grads], axis=0), grads[0].dense_shape)
    return concat_grad