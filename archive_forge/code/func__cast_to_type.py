import functools
import warnings
import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import _type_utils, errors, symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import jit_utils, registration
def _cast_to_type(g: jit_utils.GraphContext, input, to_type):
    if to_type is None:
        return input
    return getattr(opset9, f'_cast_{to_type}')(g, input, False)