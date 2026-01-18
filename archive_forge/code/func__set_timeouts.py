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
def _set_timeouts(self, **timeouts: Optional[timedelta]):
    for name, timeout in timeouts.items():
        if timeout is None:
            timeout = self._DEFAULT_TIMEOUTS[name]
        if timeout <= self._ZERO:
            raise ValueError(f'The {name} timeout ({timeout}) must be positive.')
        setattr(self, '_' + name, timeout)