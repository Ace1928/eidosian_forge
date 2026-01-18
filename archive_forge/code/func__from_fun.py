from contextlib import nullcontext
import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._functorch.aot_autograd import AOTConfig, create_joint, from_fun
from torch._higher_order_ops.cond import (
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import (
from torch.fx.experimental.proxy_tensor import (
from torch.multiprocessing.reductions import StorageWeakRef
def _from_fun(t):
    if isinstance(t, torch.Tensor):
        if t.dtype != torch.bool:
            return torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, requires_grad=t.requires_grad)
        else:
            maybe_unfunc_t = t
            if isinstance(t, FunctionalTensor):
                torch._sync(t)
                maybe_unfunc_t = from_fun(t)
            elif torch._is_functional_tensor(t):
                torch._sync(t)
                maybe_unfunc_t = torch._from_functional_tensor(t)
            return maybe_unfunc_t.clone()
    return t