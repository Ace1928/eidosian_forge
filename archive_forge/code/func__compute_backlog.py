from __future__ import annotations
import errno
import math
import sys
from typing import TYPE_CHECKING
import trio
from trio import TaskStatus
from . import socket as tsocket
from ._deprecate import warn_deprecated
def _compute_backlog(backlog: int | None) -> int:
    if backlog == math.inf:
        backlog = None
        warn_deprecated(thing='math.inf as a backlog', version='0.23.0', instead='None', issue=2842)
    if not isinstance(backlog, int) and backlog is not None:
        raise TypeError(f'backlog must be an int or None, not {backlog!r}')
    if backlog is None:
        return 65535
    return min(backlog, 65535)