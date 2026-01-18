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
def _remove_from_participants(self) -> None:
    msg = f"The node '{self._node}' removed itself from the participants of round {self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
    self._record(message=msg)
    log.debug(msg)
    state = self._state
    del state.participants[self._node]
    del state.last_heartbeats[self._node]
    _remove_participant_epilogue(state, self._settings)