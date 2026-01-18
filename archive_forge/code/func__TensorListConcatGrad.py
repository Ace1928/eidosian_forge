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
@ops.RegisterGradient('TensorListConcat')
@ops.RegisterGradient('TensorListConcatV2')
def _TensorListConcatGrad(op, dtensor, unused_dlengths):
    """Gradient function for TensorListConcat."""
    dlist = tensor_list_split(dtensor, element_shape=gen_list_ops.tensor_list_element_shape(op.inputs[0], shape_type=dtypes.int32), lengths=op.outputs[1])
    if op.type == 'TensorListConcatV2':
        return (dlist, None, None)
    else:
        return dlist