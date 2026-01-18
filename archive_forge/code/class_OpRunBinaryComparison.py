from typing import Any, Dict
import numpy as np
from onnx.onnx_pb import NodeProto
from onnx.reference.op_run import OpRun, RuntimeTypeError
class OpRunBinaryComparison(OpRunBinary):
    """Ancestor to all binary operators in this subfolder comparing tensors."""
    pass