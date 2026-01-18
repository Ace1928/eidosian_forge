import collections
import subprocess
import sys
import warnings
from . import futures
from . import protocols
from . import transports
from .coroutines import coroutine
from .log import logger
@coroutine
def _connect_pipes(self, waiter):
    try:
        proc = self._proc
        loop = self._loop
        if proc.stdin is not None:
            _, pipe = (yield from loop.connect_write_pipe(lambda: WriteSubprocessPipeProto(self, 0), proc.stdin))
            self._pipes[0] = pipe
        if proc.stdout is not None:
            _, pipe = (yield from loop.connect_read_pipe(lambda: ReadSubprocessPipeProto(self, 1), proc.stdout))
            self._pipes[1] = pipe
        if proc.stderr is not None:
            _, pipe = (yield from loop.connect_read_pipe(lambda: ReadSubprocessPipeProto(self, 2), proc.stderr))
            self._pipes[2] = pipe
        assert self._pending_calls is not None
        loop.call_soon(self._protocol.connection_made, self)
        for callback, data in self._pending_calls:
            loop.call_soon(callback, *data)
        self._pending_calls = None
    except Exception as exc:
        if waiter is not None and (not waiter.cancelled()):
            waiter.set_exception(exc)
    else:
        if waiter is not None and (not waiter.cancelled()):
            waiter.set_result(None)