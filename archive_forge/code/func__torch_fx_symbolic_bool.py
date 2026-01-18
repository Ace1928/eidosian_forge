from __future__ import annotations
import dataclasses
import functools
import logging
from typing import Any, Optional
import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_lib import graph_building  # type: ignore[import]
import torch
import torch.fx
from torch.onnx._internal import diagnostics
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import decorator, formatter
from torch.onnx._internal.fx import registration, type_utils as fx_type_utils
@_format_argument.register
def _torch_fx_symbolic_bool(obj: torch.SymBool) -> str:
    return f'SymBool({obj})'