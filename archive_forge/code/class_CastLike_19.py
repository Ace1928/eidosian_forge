from onnx.helper import np_dtype_to_tensor_dtype
from onnx.onnx_pb import TensorProto
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_cast import (
class CastLike_19(OpRun):

    def _run(self, x, y, saturate=None):
        return _cast_like(x, y, saturate)