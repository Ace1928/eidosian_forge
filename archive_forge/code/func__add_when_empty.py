from __future__ import annotations
import os
from collections import deque
from queue import Empty
from queue import LifoQueue as _LifoQueue
from typing import TYPE_CHECKING
from . import exceptions
from .utils.compat import register_after_fork
from .utils.functional import lazy
def _add_when_empty(self):
    if self.limit and len(self._dirty) >= self.limit:
        raise self.LimitExceeded(self.limit)
    self._resource.put_nowait(self.new())