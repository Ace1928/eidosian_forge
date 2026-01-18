from typing import Any, Dict
import numpy as np
from onnx.onnx_pb import NodeProto
from onnx.reference.op_run import OpRun, RuntimeTypeError
Special case reduction where the output value is a constant.