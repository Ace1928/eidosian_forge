from __future__ import annotations
import array
import asyncio
import concurrent.futures
import math
import socket
import sys
import threading
from asyncio import (
from asyncio.base_events import _run_until_complete_cb  # type: ignore[attr-defined]
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Generator, Iterable
from concurrent.futures import Future
from contextlib import suppress
from contextvars import Context, copy_context
from dataclasses import dataclass
from functools import partial, wraps
from inspect import (
from io import IOBase
from os import PathLike
from queue import Queue
from signal import Signals
from socket import AddressFamily, SocketKind
from threading import Thread
from types import TracebackType
from typing import (
from weakref import WeakKeyDictionary
import sniffio
from .. import CapacityLimiterStatistics, EventStatistics, TaskInfo, abc
from .._core._eventloop import claim_worker_thread, threadlocals
from .._core._exceptions import (
from .._core._sockets import convert_ipv6_sockaddr
from .._core._streams import create_memory_object_stream
from .._core._synchronization import CapacityLimiter as BaseCapacityLimiter
from .._core._synchronization import Event as BaseEvent
from .._core._synchronization import ResourceGuard
from .._core._tasks import CancelScope as BaseCancelScope
from ..abc import (
from ..lowlevel import RunVar
from ..streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
class AsyncIOBackend(AsyncBackend):

    @classmethod
    def run(cls, func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval]], args: tuple[Unpack[PosArgsT]], kwargs: dict[str, Any], options: dict[str, Any]) -> T_Retval:

        @wraps(func)
        async def wrapper() -> T_Retval:
            task = cast(asyncio.Task, current_task())
            task.set_name(get_callable_name(func))
            _task_states[task] = TaskState(None, None)
            try:
                return await func(*args)
            finally:
                del _task_states[task]
        debug = options.get('debug', False)
        loop_factory = options.get('loop_factory', None)
        if loop_factory is None and options.get('use_uvloop', False):
            import uvloop
            loop_factory = uvloop.new_event_loop
        with Runner(debug=debug, loop_factory=loop_factory) as runner:
            return runner.run(wrapper())

    @classmethod
    def current_token(cls) -> object:
        return get_running_loop()

    @classmethod
    def current_time(cls) -> float:
        return get_running_loop().time()

    @classmethod
    def cancelled_exception_class(cls) -> type[BaseException]:
        return CancelledError

    @classmethod
    async def checkpoint(cls) -> None:
        await sleep(0)

    @classmethod
    async def checkpoint_if_cancelled(cls) -> None:
        task = current_task()
        if task is None:
            return
        try:
            cancel_scope = _task_states[task].cancel_scope
        except KeyError:
            return
        while cancel_scope:
            if cancel_scope.cancel_called:
                await sleep(0)
            elif cancel_scope.shield:
                break
            else:
                cancel_scope = cancel_scope._parent_scope

    @classmethod
    async def cancel_shielded_checkpoint(cls) -> None:
        with CancelScope(shield=True):
            await sleep(0)

    @classmethod
    async def sleep(cls, delay: float) -> None:
        await sleep(delay)

    @classmethod
    def create_cancel_scope(cls, *, deadline: float=math.inf, shield: bool=False) -> CancelScope:
        return CancelScope(deadline=deadline, shield=shield)

    @classmethod
    def current_effective_deadline(cls) -> float:
        try:
            cancel_scope = _task_states[current_task()].cancel_scope
        except KeyError:
            return math.inf
        deadline = math.inf
        while cancel_scope:
            deadline = min(deadline, cancel_scope.deadline)
            if cancel_scope._cancel_called:
                deadline = -math.inf
                break
            elif cancel_scope.shield:
                break
            else:
                cancel_scope = cancel_scope._parent_scope
        return deadline

    @classmethod
    def create_task_group(cls) -> abc.TaskGroup:
        return TaskGroup()

    @classmethod
    def create_event(cls) -> abc.Event:
        return Event()

    @classmethod
    def create_capacity_limiter(cls, total_tokens: float) -> abc.CapacityLimiter:
        return CapacityLimiter(total_tokens)

    @classmethod
    async def run_sync_in_worker_thread(cls, func: Callable[[Unpack[PosArgsT]], T_Retval], args: tuple[Unpack[PosArgsT]], abandon_on_cancel: bool=False, limiter: abc.CapacityLimiter | None=None) -> T_Retval:
        await cls.checkpoint()
        try:
            idle_workers = _threadpool_idle_workers.get()
            workers = _threadpool_workers.get()
        except LookupError:
            idle_workers = deque()
            workers = set()
            _threadpool_idle_workers.set(idle_workers)
            _threadpool_workers.set(workers)
        async with limiter or cls.current_default_thread_limiter():
            with CancelScope(shield=not abandon_on_cancel) as scope:
                future: asyncio.Future = asyncio.Future()
                root_task = find_root_task()
                if not idle_workers:
                    worker = WorkerThread(root_task, workers, idle_workers)
                    worker.start()
                    workers.add(worker)
                    root_task.add_done_callback(worker.stop)
                else:
                    worker = idle_workers.pop()
                    now = cls.current_time()
                    while idle_workers:
                        if now - idle_workers[0].idle_since < WorkerThread.MAX_IDLE_TIME:
                            break
                        expired_worker = idle_workers.popleft()
                        expired_worker.root_task.remove_done_callback(expired_worker.stop)
                        expired_worker.stop()
                context = copy_context()
                context.run(sniffio.current_async_library_cvar.set, None)
                if abandon_on_cancel or scope._parent_scope is None:
                    worker_scope = scope
                else:
                    worker_scope = scope._parent_scope
                worker.queue.put_nowait((context, func, args, future, worker_scope))
                return await future

    @classmethod
    def check_cancelled(cls) -> None:
        scope: CancelScope | None = threadlocals.current_cancel_scope
        while scope is not None:
            if scope.cancel_called:
                raise CancelledError(f'Cancelled by cancel scope {id(scope):x}')
            if scope.shield:
                return
            scope = scope._parent_scope

    @classmethod
    def run_async_from_thread(cls, func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval]], args: tuple[Unpack[PosArgsT]], token: object) -> T_Retval:

        async def task_wrapper(scope: CancelScope) -> T_Retval:
            __tracebackhide__ = True
            task = cast(asyncio.Task, current_task())
            _task_states[task] = TaskState(None, scope)
            scope._tasks.add(task)
            try:
                return await func(*args)
            except CancelledError as exc:
                raise concurrent.futures.CancelledError(str(exc)) from None
            finally:
                scope._tasks.discard(task)
        loop = cast(AbstractEventLoop, token)
        context = copy_context()
        context.run(sniffio.current_async_library_cvar.set, 'asyncio')
        wrapper = task_wrapper(threadlocals.current_cancel_scope)
        f: concurrent.futures.Future[T_Retval] = context.run(asyncio.run_coroutine_threadsafe, wrapper, loop)
        return f.result()

    @classmethod
    def run_sync_from_thread(cls, func: Callable[[Unpack[PosArgsT]], T_Retval], args: tuple[Unpack[PosArgsT]], token: object) -> T_Retval:

        @wraps(func)
        def wrapper() -> None:
            try:
                sniffio.current_async_library_cvar.set('asyncio')
                f.set_result(func(*args))
            except BaseException as exc:
                f.set_exception(exc)
                if not isinstance(exc, Exception):
                    raise
        f: concurrent.futures.Future[T_Retval] = Future()
        loop = cast(AbstractEventLoop, token)
        loop.call_soon_threadsafe(wrapper)
        return f.result()

    @classmethod
    def create_blocking_portal(cls) -> abc.BlockingPortal:
        return BlockingPortal()

    @classmethod
    async def open_process(cls, command: str | bytes | Sequence[str | bytes], *, shell: bool, stdin: int | IO[Any] | None, stdout: int | IO[Any] | None, stderr: int | IO[Any] | None, cwd: str | bytes | PathLike | None=None, env: Mapping[str, str] | None=None, start_new_session: bool=False) -> Process:
        await cls.checkpoint()
        if shell:
            process = await asyncio.create_subprocess_shell(cast('str | bytes', command), stdin=stdin, stdout=stdout, stderr=stderr, cwd=cwd, env=env, start_new_session=start_new_session)
        else:
            process = await asyncio.create_subprocess_exec(*command, stdin=stdin, stdout=stdout, stderr=stderr, cwd=cwd, env=env, start_new_session=start_new_session)
        stdin_stream = StreamWriterWrapper(process.stdin) if process.stdin else None
        stdout_stream = StreamReaderWrapper(process.stdout) if process.stdout else None
        stderr_stream = StreamReaderWrapper(process.stderr) if process.stderr else None
        return Process(process, stdin_stream, stdout_stream, stderr_stream)

    @classmethod
    def setup_process_pool_exit_at_shutdown(cls, workers: set[abc.Process]) -> None:
        create_task(_shutdown_process_pool_on_exit(workers), name='AnyIO process pool shutdown task')
        find_root_task().add_done_callback(partial(_forcibly_shutdown_process_pool_on_exit, workers))

    @classmethod
    async def connect_tcp(cls, host: str, port: int, local_address: IPSockAddrType | None=None) -> abc.SocketStream:
        transport, protocol = cast(Tuple[asyncio.Transport, StreamProtocol], await get_running_loop().create_connection(StreamProtocol, host, port, local_addr=local_address))
        transport.pause_reading()
        return SocketStream(transport, protocol)

    @classmethod
    async def connect_unix(cls, path: str | bytes) -> abc.UNIXSocketStream:
        await cls.checkpoint()
        loop = get_running_loop()
        raw_socket = socket.socket(socket.AF_UNIX)
        raw_socket.setblocking(False)
        while True:
            try:
                raw_socket.connect(path)
            except BlockingIOError:
                f: asyncio.Future = asyncio.Future()
                loop.add_writer(raw_socket, f.set_result, None)
                f.add_done_callback(lambda _: loop.remove_writer(raw_socket))
                await f
            except BaseException:
                raw_socket.close()
                raise
            else:
                return UNIXSocketStream(raw_socket)

    @classmethod
    def create_tcp_listener(cls, sock: socket.socket) -> SocketListener:
        return TCPSocketListener(sock)

    @classmethod
    def create_unix_listener(cls, sock: socket.socket) -> SocketListener:
        return UNIXSocketListener(sock)

    @classmethod
    async def create_udp_socket(cls, family: AddressFamily, local_address: IPSockAddrType | None, remote_address: IPSockAddrType | None, reuse_port: bool) -> UDPSocket | ConnectedUDPSocket:
        transport, protocol = await get_running_loop().create_datagram_endpoint(DatagramProtocol, local_addr=local_address, remote_addr=remote_address, family=family, reuse_port=reuse_port)
        if protocol.exception:
            transport.close()
            raise protocol.exception
        if not remote_address:
            return UDPSocket(transport, protocol)
        else:
            return ConnectedUDPSocket(transport, protocol)

    @classmethod
    async def create_unix_datagram_socket(cls, raw_socket: socket.socket, remote_path: str | bytes | None) -> abc.UNIXDatagramSocket | abc.ConnectedUNIXDatagramSocket:
        await cls.checkpoint()
        loop = get_running_loop()
        if remote_path:
            while True:
                try:
                    raw_socket.connect(remote_path)
                except BlockingIOError:
                    f: asyncio.Future = asyncio.Future()
                    loop.add_writer(raw_socket, f.set_result, None)
                    f.add_done_callback(lambda _: loop.remove_writer(raw_socket))
                    await f
                except BaseException:
                    raw_socket.close()
                    raise
                else:
                    return ConnectedUNIXDatagramSocket(raw_socket)
        else:
            return UNIXDatagramSocket(raw_socket)

    @classmethod
    async def getaddrinfo(cls, host: bytes | str | None, port: str | int | None, *, family: int | AddressFamily=0, type: int | SocketKind=0, proto: int=0, flags: int=0) -> list[tuple[AddressFamily, SocketKind, int, str, tuple[str, int] | tuple[str, int, int, int]]]:
        return await get_running_loop().getaddrinfo(host, port, family=family, type=type, proto=proto, flags=flags)

    @classmethod
    async def getnameinfo(cls, sockaddr: IPSockAddrType, flags: int=0) -> tuple[str, str]:
        return await get_running_loop().getnameinfo(sockaddr, flags)

    @classmethod
    async def wait_socket_readable(cls, sock: socket.socket) -> None:
        await cls.checkpoint()
        try:
            read_events = _read_events.get()
        except LookupError:
            read_events = {}
            _read_events.set(read_events)
        if read_events.get(sock):
            raise BusyResourceError('reading from') from None
        loop = get_running_loop()
        event = read_events[sock] = asyncio.Event()
        loop.add_reader(sock, event.set)
        try:
            await event.wait()
        finally:
            if read_events.pop(sock, None) is not None:
                loop.remove_reader(sock)
                readable = True
            else:
                readable = False
        if not readable:
            raise ClosedResourceError

    @classmethod
    async def wait_socket_writable(cls, sock: socket.socket) -> None:
        await cls.checkpoint()
        try:
            write_events = _write_events.get()
        except LookupError:
            write_events = {}
            _write_events.set(write_events)
        if write_events.get(sock):
            raise BusyResourceError('writing to') from None
        loop = get_running_loop()
        event = write_events[sock] = asyncio.Event()
        loop.add_writer(sock.fileno(), event.set)
        try:
            await event.wait()
        finally:
            if write_events.pop(sock, None) is not None:
                loop.remove_writer(sock)
                writable = True
            else:
                writable = False
        if not writable:
            raise ClosedResourceError

    @classmethod
    def current_default_thread_limiter(cls) -> CapacityLimiter:
        try:
            return _default_thread_limiter.get()
        except LookupError:
            limiter = CapacityLimiter(40)
            _default_thread_limiter.set(limiter)
            return limiter

    @classmethod
    def open_signal_receiver(cls, *signals: Signals) -> ContextManager[AsyncIterator[Signals]]:
        return _SignalReceiver(signals)

    @classmethod
    def get_current_task(cls) -> TaskInfo:
        return _create_task_info(current_task())

    @classmethod
    def get_running_tasks(cls) -> list[TaskInfo]:
        return [_create_task_info(task) for task in all_tasks() if not task.done()]

    @classmethod
    async def wait_all_tasks_blocked(cls) -> None:
        await cls.checkpoint()
        this_task = current_task()
        while True:
            for task in all_tasks():
                if task is this_task:
                    continue
                waiter = task._fut_waiter
                if waiter is None or waiter.done():
                    await sleep(0.1)
                    break
            else:
                return

    @classmethod
    def create_test_runner(cls, options: dict[str, Any]) -> TestRunner:
        return TestRunner(**options)