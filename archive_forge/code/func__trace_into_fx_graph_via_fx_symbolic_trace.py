from __future__ import annotations
import functools
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.fx
import torch.onnx
import torch.onnx._internal.fx.passes as passes
from torch.onnx._internal import _beartype, exporter, io_adapter
@_beartype.beartype
def _trace_into_fx_graph_via_fx_symbolic_trace(self, model, model_args, model_kwargs) -> torch.fx.GraphModule:
    bind_input_step = io_adapter.BindInputStep(torch.onnx.utils.model_signature(model))
    self.input_adapter.append_step(bind_input_step)
    _, named_args = bind_input_step.apply(model_args, model_kwargs, model=model)
    concrete_args = {}
    for param_name, param_value in named_args.items():
        if isinstance(param_value, torch.Tensor):
            concrete_args[param_name] = torch.fx._symbolic_trace.PH
        else:
            concrete_args[param_name] = param_value
    merge_kwargs_step = io_adapter.MergeKwargsIntoArgsInputStep()
    self.input_adapter.append_step(merge_kwargs_step)
    return _module_expansion_symbolic_trace(model, concrete_args=concrete_args)