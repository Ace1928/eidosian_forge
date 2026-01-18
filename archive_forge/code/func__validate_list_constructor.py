from tensorflow.python.autograph.operators import data_structures
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_util
def _validate_list_constructor(elements, element_dtype, element_shape):
    """Validates the inputs of tensor_list."""
    if element_dtype is not None and element_shape is not None:
        return
    if tensor_util.is_tf_type(elements):
        return
    if isinstance(elements, (list, tuple)):
        if elements:
            return
        else:
            raise ValueError('element_dtype and element_shape are required when elements are empty')
    raise ValueError('unknown type for elements: {}; only Tensor, list and tuple are allowed'.format(type(elements)))