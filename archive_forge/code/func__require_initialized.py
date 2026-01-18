import collections
import contextlib
import functools
import inspect
import logging
import threading
from typing import Dict, Generic, TypeVar, Set, Any, TYPE_CHECKING
import torch
from torch.futures import Future
from torch._C._distributed_rpc import (
from .internal import (
from .constants import DEFAULT_SHUTDOWN_TIMEOUT, UNSET_RPC_TIMEOUT
from ._utils import _group_membership_management, _update_group_membership
def _require_initialized(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _is_current_rpc_agent_set():
            raise RuntimeError('RPC has not been initialized. Call torch.distributed.rpc.init_rpc first.')
        return func(*args, **kwargs)
    return wrapper