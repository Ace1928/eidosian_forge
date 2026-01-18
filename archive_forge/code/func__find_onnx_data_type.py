from __future__ import annotations
import logging
import operator
import types
from typing import (
import torch
import torch._ops
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
@_beartype.beartype
def _find_onnx_data_type(torch_input: Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list, tuple]]) -> Set[str]:
    """Convert inputs data type from torch acceptable dtype to the compatible onnx dtype string."""
    if isinstance(torch_input, fx_type_utils.TensorLike) and torch_input.dtype is not None:
        return fx_type_utils.from_torch_dtype_to_onnx_dtype_str(torch_input.dtype)
    if isinstance(torch_input, (int, float, bool, str)):
        return fx_type_utils.from_torch_dtype_to_onnx_dtype_str(type(torch_input))
    if isinstance(torch_input, (list, tuple)) and torch_input:
        set_dtype = _find_onnx_data_type(torch_input[0])
        if any((isinstance(input, fx_type_utils.TensorLike) for input in torch_input)):
            return {f'seq({dtype})' for dtype in set_dtype}
        else:
            return set_dtype
    if torch_input is None or (isinstance(torch_input, fx_type_utils.TensorLike) and torch_input.dtype is None) or (isinstance(torch_input, (list, tuple)) and (not torch_input)):
        return set()
    raise RuntimeError(f'Unknown input type from input: {torch_input}')