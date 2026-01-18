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
class RemoveInputMutation(_pass.Transform):
    """Remove `aten.copy_.default` nodes that mutate module inputs.

    This pass is recommended to be used after ``Functionalization`` pass.
    ``Functionalization`` pass adds `aten.copy_.default` nodes to the graph
    when it detects mutations to inputs. These nodes are not needed for ONNX export
    for inference. They could be useful for training.
    """

    @_beartype.beartype
    def _run(self, *args) -> torch.fx.GraphModule:
        for node in reversed(self.module.graph.nodes):
            if node.op == 'call_function' and node.target == torch.ops.aten.copy_.default and (len(node.users) == 0) and isinstance(node.args[0], torch.fx.Node) and (node.args[0].op == 'placeholder'):
                self.module.graph.erase_node(node)
        return self.module