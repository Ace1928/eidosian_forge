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
def _init_rpc_states(agent):
    worker_infos = agent.get_worker_infos()
    global _ALL_WORKER_NAMES
    _ALL_WORKER_NAMES = {worker_info.name for worker_info in worker_infos}
    if not _is_current_rpc_agent_set():
        _set_and_start_rpc_agent(agent)