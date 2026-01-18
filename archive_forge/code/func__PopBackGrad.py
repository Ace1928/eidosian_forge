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
@ops.RegisterGradient('TensorListPopBack')
def _PopBackGrad(op, dlist, delement):
    if dlist is None:
        dlist = empty_tensor_list(element_dtype=delement.dtype, element_shape=gen_list_ops.tensor_list_element_shape(op.outputs[0], shape_type=dtypes.int32))
    if delement is None:
        delement = array_ops.zeros_like(op.outputs[1])
    return (gen_list_ops.tensor_list_push_back(dlist, delement), None)