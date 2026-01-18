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
@ops.RegisterGradient('TensorListStack')
def _TensorListStackGrad(unused_op, dtensor):
    return (tensor_list_from_tensor(dtensor, element_shape=dtensor.shape[1:]), None)