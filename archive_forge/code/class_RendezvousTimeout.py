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
class RendezvousTimeout:
    """Hold the timeout configuration of a rendezvous.

    Args:
        join:
            The time within which the rendezvous is expected to complete.
        last_call:
            An additional wait amount before completing the rendezvous once the
            rendezvous has the minimum number of required participants.
        close:
            The time within which the rendezvous is expected to close after a
            call to :py:meth:`RendezvousHandler.set_closed` or
            :py:meth:`RendezvousHandler.shutdown`.
        keep_alive:
            The time within which a keep-alive heartbeat is expected to
            complete.
    """
    _ZERO = timedelta(0)
    _DEFAULT_TIMEOUTS = {'join': timedelta(seconds=600), 'last_call': timedelta(seconds=30), 'close': timedelta(seconds=30), 'heartbeat': timedelta(seconds=5)}
    _join: timedelta
    _last_call: timedelta
    _close: timedelta
    _heartbeat: timedelta

    def __init__(self, join: Optional[timedelta]=None, last_call: Optional[timedelta]=None, close: Optional[timedelta]=None, heartbeat: Optional[timedelta]=None) -> None:
        self._set_timeouts(join=join, last_call=last_call, close=close, heartbeat=heartbeat)

    @property
    def join(self) -> timedelta:
        """Get the join timeout."""
        return self._join

    @property
    def last_call(self) -> timedelta:
        """Get the last call timeout."""
        return self._last_call

    @property
    def close(self) -> timedelta:
        """Get the close timeout."""
        return self._close

    @property
    def heartbeat(self) -> timedelta:
        """Get the keep-alive heartbeat timeout."""
        return self._heartbeat

    def _set_timeouts(self, **timeouts: Optional[timedelta]):
        for name, timeout in timeouts.items():
            if timeout is None:
                timeout = self._DEFAULT_TIMEOUTS[name]
            if timeout <= self._ZERO:
                raise ValueError(f'The {name} timeout ({timeout}) must be positive.')
            setattr(self, '_' + name, timeout)