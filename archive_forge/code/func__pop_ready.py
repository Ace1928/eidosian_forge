from __future__ import annotations
import errno
import threading
from contextlib import contextmanager
from copy import copy
from queue import Empty
from time import sleep
from types import GeneratorType as generator
from vine import Thenable, promise
from kombu.log import get_logger
from kombu.utils.compat import fileno
from kombu.utils.eventio import ERR, READ, WRITE, poll
from kombu.utils.objects import cached_property
from .timer import Timer
def _pop_ready(self):
    with self._ready_lock:
        ready = self._ready
        self._ready = set()
        return ready