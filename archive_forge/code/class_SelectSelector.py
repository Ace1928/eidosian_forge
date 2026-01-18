from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Mapping
import math
import select
import sys
class SelectSelector(_BaseSelectorImpl):
    """Select-based selector."""

    def __init__(self):
        super().__init__()
        self._readers = set()
        self._writers = set()

    def register(self, fileobj, events, data=None):
        key = super().register(fileobj, events, data)
        if events & EVENT_READ:
            self._readers.add(key.fd)
        if events & EVENT_WRITE:
            self._writers.add(key.fd)
        return key

    def unregister(self, fileobj):
        key = super().unregister(fileobj)
        self._readers.discard(key.fd)
        self._writers.discard(key.fd)
        return key
    if sys.platform == 'win32':

        def _select(self, r, w, _, timeout=None):
            r, w, x = select.select(r, w, w, timeout)
            return (r, w + x, [])
    else:
        _select = select.select

    def select(self, timeout=None):
        timeout = None if timeout is None else max(timeout, 0)
        ready = []
        try:
            r, w, _ = self._select(self._readers, self._writers, [], timeout)
        except InterruptedError:
            return ready
        r = set(r)
        w = set(w)
        for fd in r | w:
            events = 0
            if fd in r:
                events |= EVENT_READ
            if fd in w:
                events |= EVENT_WRITE
            key = self._key_from_fd(fd)
            if key:
                ready.append((key, events & key.events))
        return ready