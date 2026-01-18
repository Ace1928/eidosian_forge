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
@ops.RegisterGradient('TensorListGather')
def _TensorListGatherGrad(op, dtensor):
    """Gradient function for TensorListGather."""
    input_list, indices, _ = op.inputs
    element_shape = gen_list_ops.tensor_list_element_shape(input_list, shape_type=dtypes.int32)
    num_elements = gen_list_ops.tensor_list_length(input_list)
    dlist = tensor_list_reserve(element_shape, num_elements, dtensor.dtype)
    dlist = tensor_list_scatter(tensor=dtensor, indices=indices, input_handle=dlist)
    return (dlist, None, None)