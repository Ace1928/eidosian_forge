from functools import partial
from . import functions
from . import rpc_async
import torch
from .constants import UNSET_RPC_TIMEOUT
from torch.futures import Future
def _rref_type_cont(rref_fut):
    rref_type = rref_fut.value()
    _invoke_func = _local_invoke
    bypass_type = issubclass(rref_type, torch.jit.ScriptModule) or issubclass(rref_type, torch._C.ScriptModule)
    if not bypass_type:
        func = getattr(rref_type, func_name)
        if hasattr(func, '_wrapped_async_rpc_function'):
            _invoke_func = _local_invoke_async_execution
    return rpc_api(rref.owner(), _invoke_func, args=(rref, func_name, args, kwargs), timeout=timeout)