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
@ops.RegisterGradient('TensorListResize')
def _TensorListResizeGrad(op, dlist):
    input_list, _ = op.inputs
    input_list_size = gen_list_ops.tensor_list_length(input_list)
    return (gen_list_ops.tensor_list_resize(dlist, input_list_size), None)