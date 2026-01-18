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
class _RendezvousStateHolder(ABC):
    """Hold the shared rendezvous state synced with other nodes."""

    @property
    @abstractmethod
    def state(self) -> _RendezvousState:
        """Get the local state."""

    @abstractmethod
    def sync(self) -> Optional[bool]:
        """Read or writes the latest state.

        Returns:
            A boolean value indicating whether the local state, in case marked
            as dirty, was successfully synced with other nodes.
        """

    @abstractmethod
    def mark_dirty(self) -> None:
        """Mark the local state as dirty."""