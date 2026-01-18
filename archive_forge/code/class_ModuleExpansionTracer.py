from __future__ import annotations
import functools
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.fx
import torch.onnx
import torch.onnx._internal.fx.passes as passes
from torch.onnx._internal import _beartype, exporter, io_adapter
class ModuleExpansionTracer(torch.fx._symbolic_trace.Tracer):
    """Tracer to create ONNX-exporting friendly FX graph.

    This tracer traces models into operators. That is,
    the traced graph mostly contains call_function nodes and
    has no call_module nodes. The call_module nodes
    are problematic to the use of make_fx(...) in ONNX
    exporter.
    """

    @_beartype.beartype
    def is_leaf_module(self, module: torch.nn.Module, module_qualified_name: str) -> bool:
        return False

    @_beartype.beartype
    def to_bool(self, obj: torch.fx.Proxy) -> bool:
        return False