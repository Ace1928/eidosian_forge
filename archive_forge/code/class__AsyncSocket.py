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
class _AsyncSocket(_Async, _zmq.Socket[Future]):
    _recv_futures = None
    _send_futures = None
    _state = 0
    _shadow_sock: _zmq.Socket
    _poller_class = _AsyncPoller
    _fd = None

    def __init__(self, context=None, socket_type=-1, io_loop=None, _from_socket: _zmq.Socket | None=None, **kwargs) -> None:
        if isinstance(context, _zmq.Socket):
            context, _from_socket = (None, context)
        if _from_socket is not None:
            super().__init__(shadow=_from_socket.underlying)
            self._shadow_sock = _from_socket
        else:
            super().__init__(context, socket_type, **kwargs)
            self._shadow_sock = _zmq.Socket.shadow(self.underlying)
        if io_loop is not None:
            warnings.warn(f'{self.__class__.__name__}(io_loop) argument is deprecated in pyzmq 22.2. The currently active loop will always be used.', DeprecationWarning, stacklevel=3)
        self._recv_futures = deque()
        self._send_futures = deque()
        self._state = 0
        self._fd = self._shadow_sock.FD

    @classmethod
    def from_socket(cls: type[T], socket: _zmq.Socket, io_loop: Any=None) -> T:
        """Create an async socket from an existing Socket"""
        return cls(_from_socket=socket, io_loop=io_loop)

    def close(self, linger: int | None=None) -> None:
        if not self.closed and self._fd is not None:
            event_list: list[_FutureEvent] = list(chain(self._recv_futures or [], self._send_futures or []))
            for event in event_list:
                if not event.future.done():
                    try:
                        event.future.cancel()
                    except RuntimeError:
                        pass
            self._clear_io_state()
        super().close(linger=linger)
    close.__doc__ = _zmq.Socket.close.__doc__

    def get(self, key):
        result = super().get(key)
        if key == EVENTS:
            self._schedule_remaining_events(result)
        return result
    get.__doc__ = _zmq.Socket.get.__doc__

    @overload
    def recv_multipart(self, flags: int=0, *, track: bool=False) -> Awaitable[list[bytes]]:
        ...

    @overload
    def recv_multipart(self, flags: int=0, *, copy: Literal[True], track: bool=False) -> Awaitable[list[bytes]]:
        ...

    @overload
    def recv_multipart(self, flags: int=0, *, copy: Literal[False], track: bool=False) -> Awaitable[list[_zmq.Frame]]:
        ...

    @overload
    def recv_multipart(self, flags: int=0, copy: bool=True, track: bool=False) -> Awaitable[list[bytes] | list[_zmq.Frame]]:
        ...

    def recv_multipart(self, flags: int=0, copy: bool=True, track: bool=False) -> Awaitable[list[bytes] | list[_zmq.Frame]]:
        """Receive a complete multipart zmq message.

        Returns a Future whose result will be a multipart message.
        """
        return self._add_recv_event('recv_multipart', dict(flags=flags, copy=copy, track=track))

    @overload
    def recv(self, flags: int=0, *, track: bool=False) -> Awaitable[bytes]:
        ...

    @overload
    def recv(self, flags: int=0, *, copy: Literal[True], track: bool=False) -> Awaitable[bytes]:
        ...

    @overload
    def recv(self, flags: int=0, *, copy: Literal[False], track: bool=False) -> Awaitable[_zmq.Frame]:
        ...

    def recv(self, flags: int=0, copy: bool=True, track: bool=False) -> Awaitable[bytes | _zmq.Frame]:
        """Receive a single zmq frame.

        Returns a Future, whose result will be the received frame.

        Recommend using recv_multipart instead.
        """
        return self._add_recv_event('recv', dict(flags=flags, copy=copy, track=track))

    def send_multipart(self, msg_parts: Any, flags: int=0, copy: bool=True, track=False, **kwargs) -> Awaitable[_zmq.MessageTracker | None]:
        """Send a complete multipart zmq message.

        Returns a Future that resolves when sending is complete.
        """
        kwargs['flags'] = flags
        kwargs['copy'] = copy
        kwargs['track'] = track
        return self._add_send_event('send_multipart', msg=msg_parts, kwargs=kwargs)

    def send(self, data: Any, flags: int=0, copy: bool=True, track: bool=False, **kwargs: Any) -> Awaitable[_zmq.MessageTracker | None]:
        """Send a single zmq frame.

        Returns a Future that resolves when sending is complete.

        Recommend using send_multipart instead.
        """
        kwargs['flags'] = flags
        kwargs['copy'] = copy
        kwargs['track'] = track
        kwargs.update(dict(flags=flags, copy=copy, track=track))
        return self._add_send_event('send', msg=data, kwargs=kwargs)

    def _deserialize(self, recvd, load):
        """Deserialize with Futures"""
        f = self._Future()

        def _chain(_):
            """Chain result through serialization to recvd"""
            if f.done():
                if not recvd.cancelled() and recvd.exception() is None:
                    warnings.warn(f'Future {f} completed while awaiting {recvd}. A message has been dropped!', RuntimeWarning)
                return
            if recvd.exception():
                f.set_exception(recvd.exception())
            else:
                buf = recvd.result()
                try:
                    loaded = load(buf)
                except Exception as e:
                    f.set_exception(e)
                else:
                    f.set_result(loaded)
        recvd.add_done_callback(_chain)

        def _chain_cancel(_):
            """Chain cancellation from f to recvd"""
            if recvd.done():
                return
            if f.cancelled():
                recvd.cancel()
        f.add_done_callback(_chain_cancel)
        return f

    def poll(self, timeout=None, flags=_zmq.POLLIN) -> Awaitable[int]:
        """poll the socket for events

        returns a Future for the poll results.
        """
        if self.closed:
            raise _zmq.ZMQError(_zmq.ENOTSUP)
        p = self._poller_class()
        p.register(self, flags)
        poll_future = cast(Future, p.poll(timeout))
        future = self._Future()

        def unwrap_result(f):
            if future.done():
                return
            if poll_future.cancelled():
                try:
                    future.cancel()
                except RuntimeError:
                    pass
                return
            if f.exception():
                future.set_exception(poll_future.exception())
            else:
                evts = dict(poll_future.result())
                future.set_result(evts.get(self, 0))
        if poll_future.done():
            unwrap_result(poll_future)
        else:
            poll_future.add_done_callback(unwrap_result)

        def cancel_poll(future):
            """Cancel underlying poll if request has been cancelled"""
            if not poll_future.done():
                try:
                    poll_future.cancel()
                except RuntimeError:
                    pass
        future.add_done_callback(cancel_poll)
        return future

    def recv_string(self, *args, **kwargs) -> Awaitable[str]:
        return super().recv_string(*args, **kwargs)

    def send_string(self, s: str, flags: int=0, encoding: str='utf-8') -> Awaitable[None]:
        return super().send_string(s, flags=flags, encoding=encoding)

    def _add_timeout(self, future, timeout):
        """Add a timeout for a send or recv Future"""

        def future_timeout():
            if future.done():
                return
            future.set_exception(_zmq.Again())
        return self._call_later(timeout, future_timeout)

    def _call_later(self, delay, callback):
        """Schedule a function to be called later

        Override for different IOLoop implementations

        Tornado and asyncio happen to both have ioloop.call_later
        with the same signature.
        """
        return self._get_loop().call_later(delay, callback)

    @staticmethod
    def _remove_finished_future(future, event_list, event=None):
        """Make sure that futures are removed from the event list when they resolve

        Avoids delaying cleanup until the next send/recv event,
        which may never come.
        """
        if not event_list:
            return
        try:
            event_list.remove(event)
        except ValueError:
            return

    def _add_recv_event(self, kind, kwargs=None, future=None):
        """Add a recv event, returning the corresponding Future"""
        f = future or self._Future()
        if kind.startswith('recv') and kwargs.get('flags', 0) & _zmq.DONTWAIT:
            recv = getattr(self._shadow_sock, kind)
            try:
                r = recv(**kwargs)
            except Exception as e:
                f.set_exception(e)
            else:
                f.set_result(r)
            return f
        timer = _NoTimer
        if hasattr(_zmq, 'RCVTIMEO'):
            timeout_ms = self._shadow_sock.rcvtimeo
            if timeout_ms >= 0:
                timer = self._add_timeout(f, timeout_ms * 0.001)
        _future_event = _FutureEvent(f, kind, kwargs, msg=None, timer=timer)
        self._recv_futures.append(_future_event)
        if self._shadow_sock.get(EVENTS) & POLLIN:
            self._handle_recv()
        if self._recv_futures and _future_event in self._recv_futures:
            f.add_done_callback(partial(self._remove_finished_future, event_list=self._recv_futures, event=_future_event))
            self._add_io_state(POLLIN)
        return f

    def _add_send_event(self, kind, msg=None, kwargs=None, future=None):
        """Add a send event, returning the corresponding Future"""
        f = future or self._Future()
        if kind in ('send', 'send_multipart') and (not self._send_futures):
            flags = kwargs.get('flags', 0)
            nowait_kwargs = kwargs.copy()
            nowait_kwargs['flags'] = flags | _zmq.DONTWAIT
            send = getattr(self._shadow_sock, kind)
            finish_early = True
            try:
                r = send(msg, **nowait_kwargs)
            except _zmq.Again as e:
                if flags & _zmq.DONTWAIT:
                    f.set_exception(e)
                else:
                    finish_early = False
            except Exception as e:
                f.set_exception(e)
            else:
                f.set_result(r)
            if finish_early:
                if self._recv_futures:
                    self._schedule_remaining_events()
                return f
        timer = _NoTimer
        if hasattr(_zmq, 'SNDTIMEO'):
            timeout_ms = self._shadow_sock.get(_zmq.SNDTIMEO)
            if timeout_ms >= 0:
                timer = self._add_timeout(f, timeout_ms * 0.001)
        _future_event = _FutureEvent(f, kind, kwargs=kwargs, msg=msg, timer=timer)
        self._send_futures.append(_future_event)
        f.add_done_callback(partial(self._remove_finished_future, event_list=self._send_futures, event=_future_event))
        self._add_io_state(POLLOUT)
        return f

    def _handle_recv(self):
        """Handle recv events"""
        if not self._shadow_sock.get(EVENTS) & POLLIN:
            return
        f = None
        while self._recv_futures:
            f, kind, kwargs, _, timer = self._recv_futures.popleft()
            if f.done():
                f = None
            else:
                break
        if not self._recv_futures:
            self._drop_io_state(POLLIN)
        if f is None:
            return
        timer.cancel()
        if kind == 'poll':
            f.set_result(None)
            return
        elif kind == 'recv_multipart':
            recv = self._shadow_sock.recv_multipart
        elif kind == 'recv':
            recv = self._shadow_sock.recv
        else:
            raise ValueError('Unhandled recv event type: %r' % kind)
        kwargs['flags'] |= _zmq.DONTWAIT
        try:
            result = recv(**kwargs)
        except Exception as e:
            f.set_exception(e)
        else:
            f.set_result(result)

    def _handle_send(self):
        if not self._shadow_sock.get(EVENTS) & POLLOUT:
            return
        f = None
        while self._send_futures:
            f, kind, kwargs, msg, timer = self._send_futures.popleft()
            if f.done():
                f = None
            else:
                break
        if not self._send_futures:
            self._drop_io_state(POLLOUT)
        if f is None:
            return
        timer.cancel()
        if kind == 'poll':
            f.set_result(None)
            return
        elif kind == 'send_multipart':
            send = self._shadow_sock.send_multipart
        elif kind == 'send':
            send = self._shadow_sock.send
        else:
            raise ValueError('Unhandled send event type: %r' % kind)
        kwargs['flags'] |= _zmq.DONTWAIT
        try:
            result = send(msg, **kwargs)
        except Exception as e:
            f.set_exception(e)
        else:
            f.set_result(result)

    def _handle_events(self, fd=0, events=0):
        """Dispatch IO events to _handle_recv, etc."""
        if self._shadow_sock.closed:
            return
        zmq_events = self._shadow_sock.get(EVENTS)
        if zmq_events & _zmq.POLLIN:
            self._handle_recv()
        if zmq_events & _zmq.POLLOUT:
            self._handle_send()
        self._schedule_remaining_events()

    def _schedule_remaining_events(self, events=None):
        """Schedule a call to handle_events next loop iteration

        If there are still events to handle.
        """
        if self._state == 0:
            return
        if events is None:
            events = self._shadow_sock.get(EVENTS)
        if events & self._state:
            self._call_later(0, self._handle_events)

    def _add_io_state(self, state):
        """Add io_state to poller."""
        if self._state != state:
            state = self._state = self._state | state
        self._update_handler(self._state)

    def _drop_io_state(self, state):
        """Stop poller from watching an io_state."""
        if self._state & state:
            self._state = self._state & ~state
        self._update_handler(self._state)

    def _update_handler(self, state):
        """Update IOLoop handler with state.

        zmq FD is always read-only.
        """
        if state:
            self._get_loop()
        self._schedule_remaining_events()

    def _init_io_state(self, loop=None):
        """initialize the ioloop event handler"""
        if loop is None:
            loop = self._get_loop()
        loop.add_handler(self._shadow_sock, self._handle_events, self._READ)
        self._call_later(0, self._handle_events)

    def _clear_io_state(self):
        """unregister the ioloop event handler

        called once during close
        """
        fd = self._shadow_sock
        if self._shadow_sock.closed:
            fd = self._fd
        if self._current_loop is not None:
            self._current_loop.remove_handler(fd)