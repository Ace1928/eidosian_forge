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
@ops.RegisterGradient('TensorListSplit')
def _TensorListSplitGrad(op, dlist):
    tensor, _, lengths = op.inputs
    element_shape = array_ops.slice(array_ops.shape(tensor), [1], [-1])
    element_shape = array_ops.concat([[-1], element_shape], axis=0)
    return (gen_list_ops.tensor_list_concat_v2(dlist, element_shape=element_shape, leading_dims=lengths, element_dtype=op.inputs[0].dtype)[0], None, None)