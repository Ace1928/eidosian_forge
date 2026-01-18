from __future__ import annotations
import enum
import itertools
import socket
import sys
from contextlib import contextmanager
from typing import (
import attrs
from outcome import Value
from .. import _core
from ._io_common import wake_all
from ._run import _public
from ._windows_cffi import (
@attrs.frozen(eq=False)
class _WindowsStatistics:
    tasks_waiting_read: int
    tasks_waiting_write: int
    tasks_waiting_overlapped: int
    completion_key_monitors: int
    backend: Literal['windows'] = attrs.field(init=False, default='windows')