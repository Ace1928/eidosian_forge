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
@ops.RegisterGradient('TensorListGetItem')
def _TensorListGetItemGrad(op, ditem):
    """Gradient for TensorListGetItem."""
    list_size = gen_list_ops.tensor_list_length(op.inputs[0])
    list_grad = gen_list_ops.tensor_list_set_item(gen_list_ops.tensor_list_reserve(gen_list_ops.tensor_list_element_shape(op.inputs[0], shape_type=dtypes.int32), list_size, element_dtype=ditem.dtype), index=op.inputs[1], item=ditem)
    index_grad = None
    element_shape_grad = None
    return (list_grad, index_grad, element_shape_grad)