import dataclasses
import importlib
import logging
from typing import (
from typing_extensions import TypeAlias
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx._compatibility import compatibility
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree
def _extract_graph_module_inputs(graph_module: torch.fx.GraphModule) -> Tuple[Any, ...]:
    placeholders = []
    for node in graph_module.graph.nodes:
        if node.op == 'placeholder':
            if hasattr(node, 'meta') and 'val' in node.meta:
                assert isinstance(node.meta['val'], torch.Tensor)
            placeholders.append(node)
    return tuple(placeholders)