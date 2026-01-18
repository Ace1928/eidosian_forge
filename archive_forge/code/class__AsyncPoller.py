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
class _AsyncPoller(_Async, _zmq.Poller):
    """Poller that returns a Future on poll, instead of blocking."""
    _socket_class: type[_AsyncSocket]
    _READ: int
    _WRITE: int
    raw_sockets: list[Any]

    def _watch_raw_socket(self, loop: Any, socket: Any, evt: int, f: Callable) -> None:
        """Schedule callback for a raw socket"""
        raise NotImplementedError()

    def _unwatch_raw_sockets(self, loop: Any, *sockets: Any) -> None:
        """Unschedule callback for a raw socket"""
        raise NotImplementedError()

    def poll(self, timeout=-1) -> Awaitable[list[tuple[Any, int]]]:
        """Return a Future for a poll event"""
        future = self._Future()
        if timeout == 0:
            try:
                result = super().poll(0)
            except Exception as e:
                future.set_exception(e)
            else:
                future.set_result(result)
            return future
        loop = self._get_loop()
        watcher = self._Future()
        raw_sockets: list[Any] = []

        def wake_raw(*args):
            if not watcher.done():
                watcher.set_result(None)
        watcher.add_done_callback(lambda f: self._unwatch_raw_sockets(loop, *raw_sockets))
        wrapped_sockets: list[_AsyncSocket] = []

        def _clear_wrapper_io(f):
            for s in wrapped_sockets:
                s._clear_io_state()
        for socket, mask in self.sockets:
            if isinstance(socket, _zmq.Socket):
                if not isinstance(socket, self._socket_class):
                    socket = self._socket_class.from_socket(socket)
                    wrapped_sockets.append(socket)
                if mask & _zmq.POLLIN:
                    socket._add_recv_event('poll', future=watcher)
                if mask & _zmq.POLLOUT:
                    socket._add_send_event('poll', future=watcher)
            else:
                raw_sockets.append(socket)
                evt = 0
                if mask & _zmq.POLLIN:
                    evt |= self._READ
                if mask & _zmq.POLLOUT:
                    evt |= self._WRITE
                self._watch_raw_socket(loop, socket, evt, wake_raw)

        def on_poll_ready(f):
            if future.done():
                return
            if watcher.cancelled():
                try:
                    future.cancel()
                except RuntimeError:
                    pass
                return
            if watcher.exception():
                future.set_exception(watcher.exception())
            else:
                try:
                    result = super(_AsyncPoller, self).poll(0)
                except Exception as e:
                    future.set_exception(e)
                else:
                    future.set_result(result)
        watcher.add_done_callback(on_poll_ready)
        if wrapped_sockets:
            watcher.add_done_callback(_clear_wrapper_io)
        if timeout is not None and timeout > 0:

            def trigger_timeout():
                if not watcher.done():
                    watcher.set_result(None)
            timeout_handle = loop.call_later(0.001 * timeout, trigger_timeout)

            def cancel_timeout(f):
                if hasattr(timeout_handle, 'cancel'):
                    timeout_handle.cancel()
                else:
                    loop.remove_timeout(timeout_handle)
            future.add_done_callback(cancel_timeout)

        def cancel_watcher(f):
            if not watcher.done():
                watcher.cancel()
        future.add_done_callback(cancel_watcher)
        return future