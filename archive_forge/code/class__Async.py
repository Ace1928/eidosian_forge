from __future__ import annotations
import warnings
from asyncio import Future
from collections import deque
from functools import partial
from itertools import chain
from typing import Any, Awaitable, Callable, NamedTuple, TypeVar, cast, overload
import zmq as _zmq
from zmq import EVENTS, POLLIN, POLLOUT
from zmq._typing import Literal
class _Async:
    """Mixin for common async logic"""
    _current_loop: Any = None
    _Future: type[Future]

    def _get_loop(self) -> Any:
        """Get event loop

        Notice if event loop has changed,
        and register init_io_state on activation of a new event loop
        """
        if self._current_loop is None:
            self._current_loop = self._default_loop()
            self._init_io_state(self._current_loop)
            return self._current_loop
        current_loop = self._default_loop()
        if current_loop is not self._current_loop:
            self._current_loop = current_loop
            self._init_io_state(current_loop)
        return current_loop

    def _default_loop(self) -> Any:
        raise NotImplementedError('Must be implemented in a subclass')

    def _init_io_state(self, loop=None) -> None:
        pass