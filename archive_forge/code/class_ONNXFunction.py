from __future__ import annotations
import dataclasses
import types
from typing import Optional, TYPE_CHECKING, Union
import torch._ops
from torch.onnx._internal import _beartype
@dataclasses.dataclass(frozen=True, eq=True)
class ONNXFunction:
    """A wrapper of onnx-script function.

    op_full_name: The qualified name of the function. In the form of '<namespace>::<op_name>.<overload>'.
    onnx_function: The onnx-script function from torchlib.
    is_custom: Whether the function is a custom function.
    is_complex: Whether the function is a function that handles complex valued inputs.

    """
    onnx_function: Union['onnxscript.OnnxFunction', 'onnxscript.TracedOnnxFunction']
    op_full_name: str
    is_custom: bool = False
    is_complex: bool = False