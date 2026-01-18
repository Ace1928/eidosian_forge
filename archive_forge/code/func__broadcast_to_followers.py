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
def _broadcast_to_followers(sequence_id, objects_map):
    with _all_gather_dict_lock:
        states = _all_gather_sequence_id_to_states[sequence_id]
    assert not states.proceed_signal.is_set(), f'Termination signal sequence id {sequence_id} got set twice.'
    states.gathered_objects = objects_map
    states.proceed_signal.set()