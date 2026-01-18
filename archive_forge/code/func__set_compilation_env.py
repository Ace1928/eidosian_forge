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
@contextmanager
def _set_compilation_env():
    _old_is_tracing = torch.fx._symbolic_trace._is_fx_tracing_flag
    try:
        torch.fx._symbolic_trace._is_fx_tracing_flag = False
        yield
    finally:
        torch.fx._symbolic_trace._is_fx_tracing_flag = _old_is_tracing