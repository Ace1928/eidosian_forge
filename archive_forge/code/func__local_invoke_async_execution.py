from functools import partial
from . import functions
from . import rpc_async
import torch
from .constants import UNSET_RPC_TIMEOUT
from torch.futures import Future
@functions.async_execution
def _local_invoke_async_execution(rref, func_name, args, kwargs):
    return getattr(rref.local_value(), func_name)(*args, **kwargs)