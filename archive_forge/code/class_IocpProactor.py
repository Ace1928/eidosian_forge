import sys
import _overlapped
import _winapi
import errno
import math
import msvcrt
import socket
import struct
import time
import weakref
from . import events
from . import base_subprocess
from . import futures
from . import exceptions
from . import proactor_events
from . import selector_events
from . import tasks
from . import windows_utils
from .log import logger
class IocpProactor:
    """Proactor implementation using IOCP."""

    def __init__(self, concurrency=INFINITE):
        self._loop = None
        self._results = []
        self._iocp = _overlapped.CreateIoCompletionPort(_overlapped.INVALID_HANDLE_VALUE, NULL, 0, concurrency)
        self._cache = {}
        self._registered = weakref.WeakSet()
        self._unregistered = []
        self._stopped_serving = weakref.WeakSet()

    def _check_closed(self):
        if self._iocp is None:
            raise RuntimeError('IocpProactor is closed')

    def __repr__(self):
        info = ['overlapped#=%s' % len(self._cache), 'result#=%s' % len(self._results)]
        if self._iocp is None:
            info.append('closed')
        return '<%s %s>' % (self.__class__.__name__, ' '.join(info))

    def set_loop(self, loop):
        self._loop = loop

    def select(self, timeout=None):
        if not self._results:
            self._poll(timeout)
        tmp = self._results
        self._results = []
        try:
            return tmp
        finally:
            tmp = None

    def _result(self, value):
        fut = self._loop.create_future()
        fut.set_result(value)
        return fut

    def recv(self, conn, nbytes, flags=0):
        self._register_with_iocp(conn)
        ov = _overlapped.Overlapped(NULL)
        try:
            if isinstance(conn, socket.socket):
                ov.WSARecv(conn.fileno(), nbytes, flags)
            else:
                ov.ReadFile(conn.fileno(), nbytes)
        except BrokenPipeError:
            return self._result(b'')

        def finish_recv(trans, key, ov):
            try:
                return ov.getresult()
            except OSError as exc:
                if exc.winerror in (_overlapped.ERROR_NETNAME_DELETED, _overlapped.ERROR_OPERATION_ABORTED):
                    raise ConnectionResetError(*exc.args)
                else:
                    raise
        return self._register(ov, conn, finish_recv)

    def recv_into(self, conn, buf, flags=0):
        self._register_with_iocp(conn)
        ov = _overlapped.Overlapped(NULL)
        try:
            if isinstance(conn, socket.socket):
                ov.WSARecvInto(conn.fileno(), buf, flags)
            else:
                ov.ReadFileInto(conn.fileno(), buf)
        except BrokenPipeError:
            return self._result(0)

        def finish_recv(trans, key, ov):
            try:
                return ov.getresult()
            except OSError as exc:
                if exc.winerror in (_overlapped.ERROR_NETNAME_DELETED, _overlapped.ERROR_OPERATION_ABORTED):
                    raise ConnectionResetError(*exc.args)
                else:
                    raise
        return self._register(ov, conn, finish_recv)

    def recvfrom(self, conn, nbytes, flags=0):
        self._register_with_iocp(conn)
        ov = _overlapped.Overlapped(NULL)
        try:
            ov.WSARecvFrom(conn.fileno(), nbytes, flags)
        except BrokenPipeError:
            return self._result((b'', None))

        def finish_recv(trans, key, ov):
            try:
                return ov.getresult()
            except OSError as exc:
                if exc.winerror in (_overlapped.ERROR_NETNAME_DELETED, _overlapped.ERROR_OPERATION_ABORTED):
                    raise ConnectionResetError(*exc.args)
                else:
                    raise
        return self._register(ov, conn, finish_recv)

    def recvfrom_into(self, conn, buf, flags=0):
        self._register_with_iocp(conn)
        ov = _overlapped.Overlapped(NULL)
        try:
            ov.WSARecvFromInto(conn.fileno(), buf, flags)
        except BrokenPipeError:
            return self._result((0, None))

        def finish_recv(trans, key, ov):
            try:
                return ov.getresult()
            except OSError as exc:
                if exc.winerror in (_overlapped.ERROR_NETNAME_DELETED, _overlapped.ERROR_OPERATION_ABORTED):
                    raise ConnectionResetError(*exc.args)
                else:
                    raise
        return self._register(ov, conn, finish_recv)

    def sendto(self, conn, buf, flags=0, addr=None):
        self._register_with_iocp(conn)
        ov = _overlapped.Overlapped(NULL)
        ov.WSASendTo(conn.fileno(), buf, flags, addr)

        def finish_send(trans, key, ov):
            try:
                return ov.getresult()
            except OSError as exc:
                if exc.winerror in (_overlapped.ERROR_NETNAME_DELETED, _overlapped.ERROR_OPERATION_ABORTED):
                    raise ConnectionResetError(*exc.args)
                else:
                    raise
        return self._register(ov, conn, finish_send)

    def send(self, conn, buf, flags=0):
        self._register_with_iocp(conn)
        ov = _overlapped.Overlapped(NULL)
        if isinstance(conn, socket.socket):
            ov.WSASend(conn.fileno(), buf, flags)
        else:
            ov.WriteFile(conn.fileno(), buf)

        def finish_send(trans, key, ov):
            try:
                return ov.getresult()
            except OSError as exc:
                if exc.winerror in (_overlapped.ERROR_NETNAME_DELETED, _overlapped.ERROR_OPERATION_ABORTED):
                    raise ConnectionResetError(*exc.args)
                else:
                    raise
        return self._register(ov, conn, finish_send)

    def accept(self, listener):
        self._register_with_iocp(listener)
        conn = self._get_accept_socket(listener.family)
        ov = _overlapped.Overlapped(NULL)
        ov.AcceptEx(listener.fileno(), conn.fileno())

        def finish_accept(trans, key, ov):
            ov.getresult()
            buf = struct.pack('@P', listener.fileno())
            conn.setsockopt(socket.SOL_SOCKET, _overlapped.SO_UPDATE_ACCEPT_CONTEXT, buf)
            conn.settimeout(listener.gettimeout())
            return (conn, conn.getpeername())

        async def accept_coro(future, conn):
            try:
                await future
            except exceptions.CancelledError:
                conn.close()
                raise
        future = self._register(ov, listener, finish_accept)
        coro = accept_coro(future, conn)
        tasks.ensure_future(coro, loop=self._loop)
        return future

    def connect(self, conn, address):
        if conn.type == socket.SOCK_DGRAM:
            _overlapped.WSAConnect(conn.fileno(), address)
            fut = self._loop.create_future()
            fut.set_result(None)
            return fut
        self._register_with_iocp(conn)
        try:
            _overlapped.BindLocal(conn.fileno(), conn.family)
        except OSError as e:
            if e.winerror != errno.WSAEINVAL:
                raise
            if conn.getsockname()[1] == 0:
                raise
        ov = _overlapped.Overlapped(NULL)
        ov.ConnectEx(conn.fileno(), address)

        def finish_connect(trans, key, ov):
            ov.getresult()
            conn.setsockopt(socket.SOL_SOCKET, _overlapped.SO_UPDATE_CONNECT_CONTEXT, 0)
            return conn
        return self._register(ov, conn, finish_connect)

    def sendfile(self, sock, file, offset, count):
        self._register_with_iocp(sock)
        ov = _overlapped.Overlapped(NULL)
        offset_low = offset & 4294967295
        offset_high = offset >> 32 & 4294967295
        ov.TransmitFile(sock.fileno(), msvcrt.get_osfhandle(file.fileno()), offset_low, offset_high, count, 0, 0)

        def finish_sendfile(trans, key, ov):
            try:
                return ov.getresult()
            except OSError as exc:
                if exc.winerror in (_overlapped.ERROR_NETNAME_DELETED, _overlapped.ERROR_OPERATION_ABORTED):
                    raise ConnectionResetError(*exc.args)
                else:
                    raise
        return self._register(ov, sock, finish_sendfile)

    def accept_pipe(self, pipe):
        self._register_with_iocp(pipe)
        ov = _overlapped.Overlapped(NULL)
        connected = ov.ConnectNamedPipe(pipe.fileno())
        if connected:
            return self._result(pipe)

        def finish_accept_pipe(trans, key, ov):
            ov.getresult()
            return pipe
        return self._register(ov, pipe, finish_accept_pipe)

    async def connect_pipe(self, address):
        delay = CONNECT_PIPE_INIT_DELAY
        while True:
            try:
                handle = _overlapped.ConnectPipe(address)
                break
            except OSError as exc:
                if exc.winerror != _overlapped.ERROR_PIPE_BUSY:
                    raise
            delay = min(delay * 2, CONNECT_PIPE_MAX_DELAY)
            await tasks.sleep(delay)
        return windows_utils.PipeHandle(handle)

    def wait_for_handle(self, handle, timeout=None):
        """Wait for a handle.

        Return a Future object. The result of the future is True if the wait
        completed, or False if the wait did not complete (on timeout).
        """
        return self._wait_for_handle(handle, timeout, False)

    def _wait_cancel(self, event, done_callback):
        fut = self._wait_for_handle(event, None, True)
        fut._done_callback = done_callback
        return fut

    def _wait_for_handle(self, handle, timeout, _is_cancel):
        self._check_closed()
        if timeout is None:
            ms = _winapi.INFINITE
        else:
            ms = math.ceil(timeout * 1000.0)
        ov = _overlapped.Overlapped(NULL)
        wait_handle = _overlapped.RegisterWaitWithQueue(handle, self._iocp, ov.address, ms)
        if _is_cancel:
            f = _WaitCancelFuture(ov, handle, wait_handle, loop=self._loop)
        else:
            f = _WaitHandleFuture(ov, handle, wait_handle, self, loop=self._loop)
        if f._source_traceback:
            del f._source_traceback[-1]

        def finish_wait_for_handle(trans, key, ov):
            return f._poll()
        self._cache[ov.address] = (f, ov, 0, finish_wait_for_handle)
        return f

    def _register_with_iocp(self, obj):
        if obj not in self._registered:
            self._registered.add(obj)
            _overlapped.CreateIoCompletionPort(obj.fileno(), self._iocp, 0, 0)

    def _register(self, ov, obj, callback):
        self._check_closed()
        f = _OverlappedFuture(ov, loop=self._loop)
        if f._source_traceback:
            del f._source_traceback[-1]
        if not ov.pending:
            try:
                value = callback(None, None, ov)
            except OSError as e:
                f.set_exception(e)
            else:
                f.set_result(value)
        self._cache[ov.address] = (f, ov, obj, callback)
        return f

    def _unregister(self, ov):
        """Unregister an overlapped object.

        Call this method when its future has been cancelled. The event can
        already be signalled (pending in the proactor event queue). It is also
        safe if the event is never signalled (because it was cancelled).
        """
        self._check_closed()
        self._unregistered.append(ov)

    def _get_accept_socket(self, family):
        s = socket.socket(family)
        s.settimeout(0)
        return s

    def _poll(self, timeout=None):
        if timeout is None:
            ms = INFINITE
        elif timeout < 0:
            raise ValueError('negative timeout')
        else:
            ms = math.ceil(timeout * 1000.0)
            if ms >= INFINITE:
                raise ValueError('timeout too big')
        while True:
            status = _overlapped.GetQueuedCompletionStatus(self._iocp, ms)
            if status is None:
                break
            ms = 0
            err, transferred, key, address = status
            try:
                f, ov, obj, callback = self._cache.pop(address)
            except KeyError:
                if self._loop.get_debug():
                    self._loop.call_exception_handler({'message': 'GetQueuedCompletionStatus() returned an unexpected event', 'status': 'err=%s transferred=%s key=%#x address=%#x' % (err, transferred, key, address)})
                if key not in (0, _overlapped.INVALID_HANDLE_VALUE):
                    _winapi.CloseHandle(key)
                continue
            if obj in self._stopped_serving:
                f.cancel()
            elif not f.done():
                try:
                    value = callback(transferred, key, ov)
                except OSError as e:
                    f.set_exception(e)
                    self._results.append(f)
                else:
                    f.set_result(value)
                    self._results.append(f)
                finally:
                    f = None
        for ov in self._unregistered:
            self._cache.pop(ov.address, None)
        self._unregistered.clear()

    def _stop_serving(self, obj):
        self._stopped_serving.add(obj)

    def close(self):
        if self._iocp is None:
            return
        for fut, ov, obj, callback in list(self._cache.values()):
            if fut.cancelled():
                pass
            elif isinstance(fut, _WaitCancelFuture):
                pass
            else:
                try:
                    fut.cancel()
                except OSError as exc:
                    if self._loop is not None:
                        context = {'message': 'Cancelling a future failed', 'exception': exc, 'future': fut}
                        if fut._source_traceback:
                            context['source_traceback'] = fut._source_traceback
                        self._loop.call_exception_handler(context)
        msg_update = 1.0
        start_time = time.monotonic()
        next_msg = start_time + msg_update
        while self._cache:
            if next_msg <= time.monotonic():
                logger.debug('%r is running after closing for %.1f seconds', self, time.monotonic() - start_time)
                next_msg = time.monotonic() + msg_update
            self._poll(msg_update)
        self._results = []
        _winapi.CloseHandle(self._iocp)
        self._iocp = None

    def __del__(self):
        self.close()