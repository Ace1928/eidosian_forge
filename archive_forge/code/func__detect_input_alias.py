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
def _detect_input_alias(gm):
    input_storages = set()
    for node in gm.graph.nodes:
        if node.op == 'placeholder' and 'val' in node.meta:
            input_storages.add(StorageWeakRef(node.meta['val']._typed_storage()))
        if node.op == 'output':

            def check_alias(out):
                if out is not None and 'val' in out.meta:
                    out_storage = StorageWeakRef(out.meta['val']._typed_storage())
                    return out_storage in input_storages
                return False
            if any(pytree.tree_leaves(pytree.tree_map(check_alias, node.args))):
                return True
    for _, module in gm.named_children():
        if isinstance(module, torch.fx.GraphModule) and _detect_input_alias(module):
            return True
    return False