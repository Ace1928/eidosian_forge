from __future__ import annotations
import contextlib
from typing import Callable, Optional
import torch
import torch._ops
import torch.func
import torch.fx
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics
from torch.onnx._internal.fx.passes import _utils
from torch.utils import _pytree as pytree
def _functionalize(self, function: Callable) -> Callable:

    def wrapped(*inputs):
        inputs_functional = pytree.tree_map_only(torch.Tensor, torch._to_functional_tensor, inputs)
        torch._enable_functionalization(reapply_views=True)
        try:
            out = function(*inputs_functional)
        finally:
            torch._disable_functionalization()
        flat_inputs = pytree.tree_leaves(inputs)
        flat_inputs_functional = pytree.tree_leaves(inputs_functional)
        for inpt, input_functional in zip(flat_inputs, flat_inputs_functional):
            if isinstance(input_functional, torch.Tensor):
                torch._sync(input_functional)
                inpt_new = torch._from_functional_tensor(input_functional)
        pytree.tree_map(torch._sync, out)
        out_unwrapped = pytree.tree_map(torch._from_functional_tensor, out)
        return out_unwrapped
    return wrapped