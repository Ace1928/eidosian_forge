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
class _RendezvousOpExecutor(ABC):
    """Execute rendezvous operations."""

    @abstractmethod
    def run(self, state_handler: Callable[[_RendezvousContext, float], _Action], deadline: float) -> None:
        """Execute a rendezvous operation.

        An operation is run inside a state machine and is expected to transition
        the rendezvous from one state to another.

        Args:
            state_handler:
                A callable that is expected to return the next state transition
                action based on the current state of the rendezvous.
            deadline:
                The time, in seconds, at which the operation will be considered
                timed-out.
        """