import numpy as np
from tensorflow.core.framework import full_type_pb2
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops.gen_list_ops import *
@ops.RegisterGradient('TensorListSetItem')
def _TensorListSetItemGrad(op, dlist):
    """Gradient function for TensorListSetItem."""
    input_list, index, item = op.inputs
    list_grad = gen_list_ops.tensor_list_set_item(dlist, index=index, item=array_ops.zeros_like(item))
    index_grad = None
    element_grad = tensor_list_get_item(dlist, index, element_shape=array_ops.shape(item), element_dtype=item.dtype)
    if op.get_attr('resize_if_index_out_of_bounds'):
        input_list_size = gen_list_ops.tensor_list_length(input_list)
        list_grad = gen_list_ops.tensor_list_resize(list_grad, input_list_size)
    return (list_grad, index_grad, element_grad)