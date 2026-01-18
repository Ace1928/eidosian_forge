from __future__ import annotations
import errno
import math
import select as __select__
import sys
from numbers import Integral
from . import fileno
from .compat import detect_environment
class _epoll:

    def __init__(self):
        self._epoll = epoll()

    def register(self, fd, events):
        try:
            self._epoll.register(fd, events)
        except Exception as exc:
            if getattr(exc, 'errno', None) != errno.EEXIST:
                raise
        return fd

    def unregister(self, fd):
        try:
            self._epoll.unregister(fd)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        except OSError as exc:
            if getattr(exc, 'errno', None) not in (errno.ENOENT, errno.EPERM):
                raise

    def poll(self, timeout):
        try:
            return self._epoll.poll(timeout if timeout is not None else -1)
        except Exception as exc:
            if getattr(exc, 'errno', None) != errno.EINTR:
                raise

    def close(self):
        self._epoll.close()