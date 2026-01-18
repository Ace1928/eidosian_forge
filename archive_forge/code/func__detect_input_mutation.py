from contextlib import contextmanager
from dataclasses import dataclass
import torch
import torch._subclasses.functional_tensor
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.utils import exposed_in
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import _get_current_dispatch_mode
def _detect_input_mutation(gm):
    input_nodes = set()
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            input_nodes.add(node)
        if node.op == 'call_function':
            target = node.target
            if isinstance(target, torch._ops.OpOverload) and target._schema.is_mutable:
                for arg in node.args:
                    if arg in input_nodes:
                        return True
    for _, module in gm.named_children():
        if isinstance(module, torch.fx.GraphModule):
            if _detect_input_mutation(module):
                return True
    return False