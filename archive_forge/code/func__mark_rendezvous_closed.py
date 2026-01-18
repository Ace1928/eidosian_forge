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
def _mark_rendezvous_closed(self) -> None:
    msg = f"The node '{self._node}' marked the rendezvous '{self._settings.run_id}' as closed. Pending sync."
    self._record(message=msg, node_state=NodeState.SUCCEEDED)
    log.debug(msg)
    self._state.closed = True