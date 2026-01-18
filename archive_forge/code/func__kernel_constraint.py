from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.tools.docs import doc_controls
def _kernel_constraint(self, kernel):
    """Radially constraints a kernel with shape (height, width, channels)."""
    padding = backend.constant([[1, 1], [1, 1]], dtype='int32')
    kernel_shape = backend.shape(kernel)[0]
    start = backend.cast(kernel_shape / 2, 'int32')
    kernel_new = backend.switch(backend.cast(math_ops.floormod(kernel_shape, 2), 'bool'), lambda: kernel[start - 1:start, start - 1:start], lambda: kernel[start - 1:start, start - 1:start] + backend.zeros((2, 2), dtype=kernel.dtype))
    index = backend.switch(backend.cast(math_ops.floormod(kernel_shape, 2), 'bool'), lambda: backend.constant(0, dtype='int32'), lambda: backend.constant(1, dtype='int32'))
    while_condition = lambda index, *args: backend.less(index, start)

    def body_fn(i, array):
        return (i + 1, array_ops.pad(array, padding, constant_values=kernel[start + i, start + i]))
    _, kernel_new = while_loop.while_loop(while_condition, body_fn, [index, kernel_new], shape_invariants=[index.get_shape(), tensor_shape.TensorShape([None, None])])
    return kernel_new