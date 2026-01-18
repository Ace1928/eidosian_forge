from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import onnxscript  # type: ignore[import]
from onnxscript import evaluator  # type: ignore[import]
import torch
import torch.fx
from torch.fx.experimental import symbolic_shapes
from torch.onnx import _constants, _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from torch.utils import _pytree
@_beartype.beartype
def _convert_tensor_to_numpy(input: fx_type_utils.Argument) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(f"{__name__} needs numpy, but it's not installed.") from exc
    if isinstance(input, torch.Tensor):
        if torch.is_complex(input):
            input = torch.view_as_real(input.resolve_conj())
        return input.detach().cpu().numpy()
    if isinstance(input, torch.dtype):
        return int(jit_type_utils.JitScalarType.from_dtype(input).onnx_type())
    if isinstance(input, (tuple, list)):
        if len(input) == 0:
            return np.array((), dtype=np.int64)
        if isinstance(input[0], torch.Tensor):
            return [_convert_tensor_to_numpy(x) for x in input]
        if isinstance(input[0], bool):
            return np.array(input, dtype=np.bool_)
        if isinstance(input[0], int):
            return np.array(input, dtype=np.int64)
        if isinstance(input[0], float):
            return np.array(input)
    return input