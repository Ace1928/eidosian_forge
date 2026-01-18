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
def _find_operator_overloads_in_onnx_registry_disagnostic_message_formatter(fn: Callable, self, node: torch.fx.Node, *args, **kwargs) -> str:
    """Format the diagnostic message for the nearest match warning."""
    return f"Searching operator overload: '{node.target}' in onnx registry...\n"