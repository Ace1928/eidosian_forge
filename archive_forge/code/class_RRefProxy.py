from functools import partial
from . import functions
from . import rpc_async
import torch
from .constants import UNSET_RPC_TIMEOUT
from torch.futures import Future
class RRefProxy:

    def __init__(self, rref, rpc_api, timeout=UNSET_RPC_TIMEOUT):
        self.rref = rref
        self.rpc_api = rpc_api
        self.rpc_timeout = timeout

    def __getattr__(self, func_name):
        return partial(_invoke_rpc, self.rref, self.rpc_api, func_name, self.rpc_timeout)