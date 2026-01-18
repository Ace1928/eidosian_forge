import sys
import os
import threading
import collections
import time
import types
import weakref
import errno
from queue import Empty, Full
from . import connection
from . import context
from .util import debug, info, Finalize, register_after_fork, is_exiting
class SimpleQueue(object):

    def __init__(self, *, ctx):
        self._reader, self._writer = connection.Pipe(duplex=False)
        self._rlock = ctx.Lock()
        self._poll = self._reader.poll
        if sys.platform == 'win32':
            self._wlock = None
        else:
            self._wlock = ctx.Lock()

    def close(self):
        self._reader.close()
        self._writer.close()

    def empty(self):
        return not self._poll()

    def __getstate__(self):
        context.assert_spawning(self)
        return (self._reader, self._writer, self._rlock, self._wlock)

    def __setstate__(self, state):
        self._reader, self._writer, self._rlock, self._wlock = state
        self._poll = self._reader.poll

    def get(self):
        with self._rlock:
            res = self._reader.recv_bytes()
        return _ForkingPickler.loads(res)

    def put(self, obj):
        obj = _ForkingPickler.dumps(obj)
        if self._wlock is None:
            self._writer.send_bytes(obj)
        else:
            with self._wlock:
                self._writer.send_bytes(obj)
    __class_getitem__ = classmethod(types.GenericAlias)