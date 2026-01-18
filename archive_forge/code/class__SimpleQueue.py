import sys
import os
import threading
import collections
import weakref
import errno
from . import connection
from . import context
from .compat import get_errno
from time import monotonic
from queue import Empty, Full
from .util import (
from .reduction import ForkingPickler
class _SimpleQueue:
    """
    Simplified Queue type -- really just a locked pipe
    """

    def __init__(self, rnonblock=False, wnonblock=False, ctx=None):
        self._reader, self._writer = connection.Pipe(duplex=False, rnonblock=rnonblock, wnonblock=wnonblock)
        self._poll = self._reader.poll
        self._rlock = self._wlock = None

    def empty(self):
        return not self._poll()

    def __getstate__(self):
        context.assert_spawning(self)
        return (self._reader, self._writer, self._rlock, self._wlock)

    def __setstate__(self, state):
        self._reader, self._writer, self._rlock, self._wlock = state

    def get_payload(self):
        return self._reader.recv_bytes()

    def send_payload(self, value):
        self._writer.send_bytes(value)

    def get(self):
        return ForkingPickler.loads(self.get_payload())

    def put(self, obj):
        self.send_payload(ForkingPickler.dumps(obj))

    def close(self):
        if self._reader is not None:
            try:
                self._reader.close()
            finally:
                self._reader = None
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None