from typing import Optional
import numpy as np
from onnx import TensorProto
from onnx.helper import np_dtype_to_tensor_dtype
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32
from onnx.reference.custom_element_types import (
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_quantize_linear import reshape_input
class DequantizeLinear_19(_CommonDequantizeLinear):

    def _run(self, x, x_scale, x_zero_point=None, axis=None):
        if len(x_scale.shape) > 1:
            raise ValueError('Input 2 must be a vector or a number.')
        return super()._run(x, x_scale, x_zero_point, axis)