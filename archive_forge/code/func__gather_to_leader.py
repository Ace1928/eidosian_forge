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
def _gather_to_leader(sequence_id, worker_name, obj, worker_names=None):
    with _all_gather_dict_lock:
        if not worker_names:
            worker_names = _ALL_WORKER_NAMES
            assert worker_name in worker_names, f'{worker_name} is not expected by leader.'
        states = _all_gather_sequence_id_to_states[sequence_id]
        assert worker_name not in states.gathered_objects, f'{worker_name} reported intent sequence id {sequence_id} twice. '
        states.gathered_objects[worker_name] = obj
        if worker_names == set(states.gathered_objects.keys()):
            states.proceed_signal.set()