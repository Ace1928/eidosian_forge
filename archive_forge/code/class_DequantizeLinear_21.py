from typing import Optional
import numpy as np
from onnx import TensorProto
from onnx.helper import np_dtype_to_tensor_dtype
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32
from onnx.reference.custom_element_types import (
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_quantize_linear import reshape_input
class DequantizeLinear_21(_CommonDequantizeLinear):

    def _run(self, *args, axis=None, block_size=None):
        return super()._run(*args, axis=axis, block_size=block_size)