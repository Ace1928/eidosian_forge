import inspect
import logging
import os
import pickle
import socket
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast
from torch.distributed import PrefixStore, Store
from torch.distributed.elastic.events import (
from .api import (
from .utils import _delay, _PeriodicTimer
class _RendezvousJoinOp:
    """Represent a rendezvous join operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        state = ctx.state
        if state.closed:
            return _Action.ERROR_CLOSED
        is_participant = ctx.node in state.participants
        if state.complete and is_participant:
            return _Action.FINISH
        now = time.monotonic()
        if now > deadline:
            rollback_period = 5
            if now <= deadline + rollback_period:
                if is_participant:
                    return _Action.REMOVE_FROM_PARTICIPANTS
                if ctx.node in state.wait_list:
                    return _Action.REMOVE_FROM_WAIT_LIST
            return _Action.ERROR_TIMEOUT
        if state.complete:
            if len(state.participants) < ctx.settings.max_nodes:
                if ctx.node not in state.wait_list:
                    return _Action.ADD_TO_WAIT_LIST
        elif is_participant:
            if len(state.participants) >= ctx.settings.min_nodes:
                if cast(datetime, state.deadline) < datetime.utcnow():
                    return _Action.MARK_RENDEZVOUS_COMPLETE
        else:
            return _Action.ADD_TO_PARTICIPANTS
        if _should_keep_alive(ctx):
            return _Action.KEEP_ALIVE
        return _Action.SYNC