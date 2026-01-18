from onnx.helper import np_dtype_to_tensor_dtype
from onnx.onnx_pb import TensorProto
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_cast import (
def _cast_like(x, y, saturate):
    if y.dtype == bfloat16 and y.dtype.descr[0][0] == 'bfloat16':
        to = TensorProto.BFLOAT16
    elif y.dtype == float8e4m3fn and y.dtype.descr[0][0] == 'e4m3fn':
        to = TensorProto.FLOAT8E4M3FN
    elif y.dtype == float8e4m3fnuz and y.dtype.descr[0][0] == 'e4m3fnuz':
        to = TensorProto.FLOAT8E4M3FNUZ
    elif y.dtype == float8e5m2 and y.dtype.descr[0][0] == 'e5m2':
        to = TensorProto.FLOAT8E5M2
    elif y.dtype == float8e5m2fnuz and y.dtype.descr[0][0] == 'e5m2fnuz':
        to = TensorProto.FLOAT8E5M2FNUZ
    elif y.dtype == uint4 and y.dtype.descr[0][0] == 'uint4':
        to = TensorProto.UINT4
    elif y.dtype == int4 and y.dtype.descr[0][0] == 'int4':
        to = TensorProto.INT4
    else:
        to = np_dtype_to_tensor_dtype(y.dtype)
    return (cast_to(x, to, saturate),)