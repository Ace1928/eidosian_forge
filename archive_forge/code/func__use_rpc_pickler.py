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
@contextlib.contextmanager
def _use_rpc_pickler(rpc_pickler):
    """
    rpc_pickler: (.internal._InternalRPCPickler) Overrides the default RPC pickler
    """
    global _default_pickler
    _default_pickler = rpc_pickler
    try:
        yield
    finally:
        _default_pickler = _internal_rpc_pickler