from __future__ import annotations
import logging # isort:skip
import inspect
import time
from copy import copy
from functools import wraps
from typing import (
from tornado import locks
from ..events import ConnectionLost
from ..util.token import generate_jwt_token
from .callbacks import DocumentCallbackGroup
def _needs_document_lock(func: F) -> F:
    """Decorator that adds the necessary locking and post-processing
       to manipulate the session's document. Expects to decorate a
       method on ServerSession and transforms it into a coroutine
       if it wasn't already.
    """

    @wraps(func)
    async def _needs_document_lock_wrapper(self: ServerSession, *args, **kwargs):
        if self.destroyed:
            log.debug('Ignoring locked callback on already-destroyed session.')
            return None
        self.block_expiration()
        try:
            with await self._lock.acquire():
                if self._pending_writes is not None:
                    raise RuntimeError('internal class invariant violated: _pending_writes ' + 'should be None if lock is not held')
                self._pending_writes = []
                try:
                    result = func(self, *args, **kwargs)
                    if inspect.isawaitable(result):
                        result = await result
                finally:
                    pending_writes = self._pending_writes
                    self._pending_writes = None
                for p in pending_writes:
                    await p
            return result
        finally:
            self.unblock_expiration()
    return _needs_document_lock_wrapper