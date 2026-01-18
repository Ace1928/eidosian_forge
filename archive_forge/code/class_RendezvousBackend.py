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
class RendezvousBackend(ABC):
    """Represent a backend that holds the rendezvous state."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the backend."""

    @abstractmethod
    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        """Get the rendezvous state.

        Returns:
            A tuple of the encoded rendezvous state and its fencing token or
            ``None`` if no state is found in the backend.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """

    @abstractmethod
    def set_state(self, state: bytes, token: Optional[Token]=None) -> Optional[Tuple[bytes, Token, bool]]:
        """Set the rendezvous state.

        The new rendezvous state is set conditionally:

          - If the specified ``token`` matches the fencing token stored in the
            backend, the state will be updated. The new state will be returned
            to the caller along with its fencing token.
          - If the specified ``token`` does not match the fencing token stored
            in the backend, the state won't be updated; instead the existing
            state along with its fencing token will be returned to the caller.
          - If the specified ``token`` is ``None``, the new state will be set
            only if there is no existing state in the backend. Either the new
            state or the existing state along with its fencing token will be
            returned to the caller.

        Args:
            state:
                The encoded rendezvous state.
            token:
                An optional fencing token that was retrieved by a previous call
                to :py:meth:`get_state` or ``set_state()``.

        Returns:
            A tuple of the serialized rendezvous state, its fencing token, and
            a boolean value indicating whether our set attempt succeeded.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """