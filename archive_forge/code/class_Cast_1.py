import numpy as np
from onnx import subbyte
from onnx.helper import (
from onnx.numpy_helper import (
from onnx.onnx_pb import TensorProto
from onnx.reference.custom_element_types import (
from onnx.reference.op_run import OpRun
class Cast_1(OpRun):

    def _run(self, x, to=None):
        return (cast_to(x, to, saturate=True),)