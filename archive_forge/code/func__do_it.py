import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
def _do_it(self, start_fn, *args):
    if not self._hooks_inited:
        self._hooks_inited = True
        firstiter, self._finalizer = get_asyncgen_hooks()
        if firstiter is not None:
            firstiter(self)
        if sys.implementation.name == 'pypy':
            self._pypy_issue2786_workaround.add(self._coroutine)
    if getcoroutinestate(self._coroutine) is CORO_CLOSED:
        raise StopAsyncIteration()

    async def step():
        if self.ag_running:
            raise ValueError('async generator already executing')
        try:
            self.ag_running = True
            return await ANextIter(self._it, start_fn, *args)
        except StopAsyncIteration:
            self._pypy_issue2786_workaround.discard(self._coroutine)
            raise
        finally:
            self.ag_running = False
    return step()