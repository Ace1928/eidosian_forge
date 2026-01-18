from __future__ import annotations
import inspect
import logging
import operator
import re
import types
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
import torch
import torch.fx
from torch.onnx import _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from torch.utils import _pytree
@_beartype.beartype
def _wrap_fx_args_as_onnxscript_args(complete_args: List[fx_type_utils.Argument], complete_kwargs: Dict[str, fx_type_utils.Argument], fx_name_to_onnxscript_value: Dict[str, Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]]], tracer: onnxscript_graph_building.TorchScriptTracingEvaluator) -> Tuple[Sequence[Optional[Union[onnxscript_graph_building.TorchScriptTensor, str, int, float, bool, list]]], Dict[str, fx_type_utils.Argument]]:
    """Map all FX arguments of a node to arguments in TorchScript graph."""
    onnxscript_args = tuple((_retrieve_or_adapt_input_to_graph_set(arg, fx_name_to_onnxscript_value, tracer) for arg in complete_args))
    onnxscript_kwargs = filter_incompatible_and_dtype_convert_kwargs(complete_kwargs)
    return (onnxscript_args, onnxscript_kwargs)