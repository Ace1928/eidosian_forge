from functools import partial
from . import functions
from . import rpc_async
import torch
from .constants import UNSET_RPC_TIMEOUT
from torch.futures import Future
def _wrap_rref_type_cont(fut):
    try:
        _rref_type_cont(fut).then(_complete_op)
    except BaseException as ex:
        result.set_exception(ex)