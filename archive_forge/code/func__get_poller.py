from __future__ import annotations
import errno
import math
import select as __select__
import sys
from numbers import Integral
from . import fileno
from .compat import detect_environment
def _get_poller():
    if detect_environment() != 'default':
        return _select
    elif epoll:
        return _epoll
    elif kqueue and 'netbsd' in sys.platform:
        return _kqueue
    elif xpoll:
        return _poll
    else:
        return _select