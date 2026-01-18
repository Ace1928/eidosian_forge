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
@ops.RegisterGradient('TensorListScatterIntoExistingList')
def _TensorListScatterIntoExistingListGrad(op, dlist):
    """Gradient function for TensorListScatterIntoExistingList."""
    _, tensor, indices = op.inputs
    dtensor = gen_list_ops.tensor_list_gather(dlist, indices, element_shape=array_ops.slice(array_ops.shape(tensor), [1], [-1]), element_dtype=tensor.dtype)
    zeros = array_ops.zeros_like(tensor)
    dlist = tensor_list_scatter(zeros, indices, indices, input_handle=dlist)
    return (dlist, dtensor, None)