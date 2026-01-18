from __future__ import annotations
import functools
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.fx
import torch.onnx
import torch.onnx._internal.fx.passes as passes
from torch.onnx._internal import _beartype, exporter, io_adapter
@_beartype.beartype
def _module_expansion_symbolic_trace(root: Union[torch.nn.Module, Callable[..., Any]], concrete_args: Optional[Dict[str, Any]]=None) -> torch.fx.GraphModule:
    """Trace a callable into FX graph.

    When "root" is torch.nn.Module, calls to its submodule (type: torch.nn.Module) will be
    expanded into operators (e.g., torch.matmul, torch.add, +, and -) to simplify graph
    structure.
    """
    patched_torch_methods = {target_name: _wrap_for_symbolic_trace(getattr(torch, target_name)) for target_name in _TORCH_METHODS_TO_PATCH}
    for name, (wrapper, _) in patched_torch_methods.items():
        setattr(torch, name, wrapper)
    try:
        tracer = ModuleExpansionTracer()
        graph = tracer.trace(root, concrete_args)
        name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
        return torch.fx.GraphModule(tracer.root, graph, name)
    finally:
        for name, (_, wrapped) in patched_torch_methods.items():
            setattr(torch, name, wrapped)