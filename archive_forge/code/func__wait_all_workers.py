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
@_require_initialized
def _wait_all_workers(timeout=DEFAULT_SHUTDOWN_TIMEOUT):
    """
    Block until all local and remote RPC processes reach this method and wait
    for all outstanding work to complete. Every RPC process must call this
    method before exit to perform a graceful shutdown. This should be used to
    terminate the RPC framework, and there is no guarantee that the RPC
    framework will work after this method returns.
    """
    try:
        _all_gather(None, timeout=timeout)
    except RuntimeError as ex:
        logger.error("Failed to respond to 'Shutdown Proceed' in time, got error %s", ex)
        raise ex