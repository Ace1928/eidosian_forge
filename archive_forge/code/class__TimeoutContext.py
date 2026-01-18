from __future__ import annotations
import functools
import time
from collections import deque
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import Any, Callable, Deque, MutableMapping, Optional, TypeVar, cast
from pymongo.write_concern import WriteConcern
class _TimeoutContext(AbstractContextManager):
    """Internal timeout context manager.

    Use :func:`pymongo.timeout` instead::

      with pymongo.timeout(0.5):
          client.test.test.insert_one({})
    """

    def __init__(self, timeout: Optional[float]):
        self._timeout = timeout
        self._tokens: Optional[tuple[Token[Optional[float]], Token[float], Token[float]]] = None

    def __enter__(self) -> _TimeoutContext:
        timeout_token = TIMEOUT.set(self._timeout)
        prev_deadline = DEADLINE.get()
        next_deadline = time.monotonic() + self._timeout if self._timeout else float('inf')
        deadline_token = DEADLINE.set(min(prev_deadline, next_deadline))
        rtt_token = RTT.set(0.0)
        self._tokens = (timeout_token, deadline_token, rtt_token)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._tokens:
            timeout_token, deadline_token, rtt_token = self._tokens
            TIMEOUT.reset(timeout_token)
            DEADLINE.reset(deadline_token)
            RTT.reset(rtt_token)