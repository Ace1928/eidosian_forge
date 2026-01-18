from __future__ import annotations
import os
import select
import socket as _stdlib_socket
import sys
from operator import index
from socket import AddressFamily, SocketKind
from typing import (
import idna as _idna
import trio
from trio._util import wraps as _wraps
from . import _core
class _SocketType(SocketType):

    def __init__(self, sock: _stdlib_socket.socket):
        if type(sock) is not _stdlib_socket.socket:
            raise TypeError(f"expected object of type 'socket.socket', not '{type(sock).__name__}'")
        self._sock = sock
        self._sock.setblocking(False)
        self._did_shutdown_SHUT_WR = False

    def detach(self) -> int:
        return self._sock.detach()

    def fileno(self) -> int:
        return self._sock.fileno()

    def getpeername(self) -> AddressFormat:
        return self._sock.getpeername()

    def getsockname(self) -> AddressFormat:
        return self._sock.getsockname()

    @overload
    def getsockopt(self, /, level: int, optname: int) -> int:
        ...

    @overload
    def getsockopt(self, /, level: int, optname: int, buflen: int) -> bytes:
        ...

    def getsockopt(self, /, level: int, optname: int, buflen: int | None=None) -> int | bytes:
        if buflen is None:
            return self._sock.getsockopt(level, optname)
        return self._sock.getsockopt(level, optname, buflen)

    @overload
    def setsockopt(self, /, level: int, optname: int, value: int | Buffer) -> None:
        ...

    @overload
    def setsockopt(self, /, level: int, optname: int, value: None, optlen: int) -> None:
        ...

    def setsockopt(self, /, level: int, optname: int, value: int | Buffer | None, optlen: int | None=None) -> None:
        if optlen is None:
            if value is None:
                raise TypeError("invalid value for argument 'value', must not be None when specifying optlen")
            return self._sock.setsockopt(level, optname, value)
        if value is not None:
            raise TypeError(f"invalid value for argument 'value': {value!r}, must be None when specifying optlen")
        return self._sock.setsockopt(level, optname, value, optlen)

    def listen(self, /, backlog: int=min(_stdlib_socket.SOMAXCONN, 128)) -> None:
        return self._sock.listen(backlog)

    def get_inheritable(self) -> bool:
        return self._sock.get_inheritable()

    def set_inheritable(self, inheritable: bool) -> None:
        return self._sock.set_inheritable(inheritable)
    if sys.platform == 'win32' or (not TYPE_CHECKING and hasattr(_stdlib_socket.socket, 'share')):

        def share(self, /, process_id: int) -> bytes:
            return self._sock.share(process_id)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        return self._sock.__exit__(exc_type, exc_value, traceback)

    @property
    def family(self) -> AddressFamily:
        return self._sock.family

    @property
    def type(self) -> SocketKind:
        return self._sock.type

    @property
    def proto(self) -> int:
        return self._sock.proto

    @property
    def did_shutdown_SHUT_WR(self) -> bool:
        return self._did_shutdown_SHUT_WR

    def __repr__(self) -> str:
        return repr(self._sock).replace('socket.socket', 'trio.socket.socket')

    def dup(self) -> SocketType:
        """Same as :meth:`socket.socket.dup`."""
        return _SocketType(self._sock.dup())

    def close(self) -> None:
        if self._sock.fileno() != -1:
            trio.lowlevel.notify_closing(self._sock)
            self._sock.close()

    async def bind(self, address: AddressFormat) -> None:
        address = await self._resolve_address_nocp(address, local=True)
        if hasattr(_stdlib_socket, 'AF_UNIX') and self.family == _stdlib_socket.AF_UNIX and address[0]:
            return await trio.to_thread.run_sync(self._sock.bind, address)
        else:
            await trio.lowlevel.checkpoint()
            return self._sock.bind(address)

    def shutdown(self, flag: int) -> None:
        self._sock.shutdown(flag)
        if flag in [_stdlib_socket.SHUT_WR, _stdlib_socket.SHUT_RDWR]:
            self._did_shutdown_SHUT_WR = True

    def is_readable(self) -> bool:
        if sys.platform == 'win32':
            rready, _, _ = select.select([self._sock], [], [], 0)
            return bool(rready)
        p = select.poll()
        p.register(self._sock, select.POLLIN)
        return bool(p.poll(0))

    async def wait_writable(self) -> None:
        await _core.wait_writable(self._sock)

    async def _resolve_address_nocp(self, address: AddressFormat, *, local: bool) -> AddressFormat:
        if self.family == _stdlib_socket.AF_INET6:
            ipv6_v6only = self._sock.getsockopt(_stdlib_socket.IPPROTO_IPV6, _stdlib_socket.IPV6_V6ONLY)
        else:
            ipv6_v6only = False
        return await _resolve_address_nocp(self.type, self.family, self.proto, ipv6_v6only=ipv6_v6only, address=address, local=local)

    async def _nonblocking_helper(self, wait_fn: Callable[[_stdlib_socket.socket], Awaitable[None]], fn: Callable[Concatenate[_stdlib_socket.socket, P], T], *args: P.args, **kwargs: P.kwargs) -> T:
        async with _try_sync():
            return fn(self._sock, *args, **kwargs)
        while True:
            await wait_fn(self._sock)
            try:
                return fn(self._sock, *args, **kwargs)
            except BlockingIOError:
                pass
    _accept = _make_simple_sock_method_wrapper(_stdlib_socket.socket.accept, _core.wait_readable)

    async def accept(self) -> tuple[SocketType, AddressFormat]:
        """Like :meth:`socket.socket.accept`, but async."""
        sock, addr = await self._accept()
        return (from_stdlib_socket(sock), addr)

    async def connect(self, address: AddressFormat) -> None:
        try:
            address = await self._resolve_address_nocp(address, local=False)
            async with _try_sync():
                return self._sock.connect(address)
            await _core.wait_writable(self._sock)
        except trio.Cancelled:
            self._sock.close()
            raise
        err = self._sock.getsockopt(_stdlib_socket.SOL_SOCKET, _stdlib_socket.SO_ERROR)
        if err != 0:
            raise OSError(err, f'Error connecting to {address!r}: {os.strerror(err)}')
    if TYPE_CHECKING:

        def recv(__self, __buflen: int, __flags: int=0) -> Awaitable[bytes]:
            ...
    recv = _make_simple_sock_method_wrapper(_stdlib_socket.socket.recv, _core.wait_readable)
    if TYPE_CHECKING:

        def recv_into(__self, buffer: Buffer, nbytes: int=0, flags: int=0) -> Awaitable[int]:
            ...
    recv_into = _make_simple_sock_method_wrapper(_stdlib_socket.socket.recv_into, _core.wait_readable)
    if TYPE_CHECKING:

        def recvfrom(__self, __bufsize: int, __flags: int=0) -> Awaitable[tuple[bytes, AddressFormat]]:
            ...
    recvfrom = _make_simple_sock_method_wrapper(_stdlib_socket.socket.recvfrom, _core.wait_readable)
    if TYPE_CHECKING:

        def recvfrom_into(__self, buffer: Buffer, nbytes: int=0, flags: int=0) -> Awaitable[tuple[int, AddressFormat]]:
            ...
    recvfrom_into = _make_simple_sock_method_wrapper(_stdlib_socket.socket.recvfrom_into, _core.wait_readable)
    if sys.platform != 'win32' or (not TYPE_CHECKING and hasattr(_stdlib_socket.socket, 'recvmsg')):
        if TYPE_CHECKING:

            def recvmsg(__self, __bufsize: int, __ancbufsize: int=0, __flags: int=0) -> Awaitable[tuple[bytes, list[tuple[int, int, bytes]], int, Any]]:
                ...
        recvmsg = _make_simple_sock_method_wrapper(_stdlib_socket.socket.recvmsg, _core.wait_readable, maybe_avail=True)
    if sys.platform != 'win32' or (not TYPE_CHECKING and hasattr(_stdlib_socket.socket, 'recvmsg_into')):
        if TYPE_CHECKING:

            def recvmsg_into(__self, __buffers: Iterable[Buffer], __ancbufsize: int=0, __flags: int=0) -> Awaitable[tuple[int, list[tuple[int, int, bytes]], int, Any]]:
                ...
        recvmsg_into = _make_simple_sock_method_wrapper(_stdlib_socket.socket.recvmsg_into, _core.wait_readable, maybe_avail=True)
    if TYPE_CHECKING:

        def send(__self, __bytes: Buffer, __flags: int=0) -> Awaitable[int]:
            ...
    send = _make_simple_sock_method_wrapper(_stdlib_socket.socket.send, _core.wait_writable)

    @overload
    async def sendto(self, __data: Buffer, __address: tuple[object, ...] | str | Buffer) -> int:
        ...

    @overload
    async def sendto(self, __data: Buffer, __flags: int, __address: tuple[object, ...] | str | Buffer) -> int:
        ...

    @_wraps(_stdlib_socket.socket.sendto, assigned=(), updated=())
    async def sendto(self, *args: Any) -> int:
        """Similar to :meth:`socket.socket.sendto`, but async."""
        args_list = list(args)
        args_list[-1] = await self._resolve_address_nocp(args[-1], local=False)
        return await self._nonblocking_helper(_core.wait_writable, _stdlib_socket.socket.sendto, *args_list)
    if sys.platform != 'win32' or (not TYPE_CHECKING and hasattr(_stdlib_socket.socket, 'sendmsg')):

        @_wraps(_stdlib_socket.socket.sendmsg, assigned=(), updated=())
        async def sendmsg(self, __buffers: Iterable[Buffer], __ancdata: Iterable[tuple[int, int, Buffer]]=(), __flags: int=0, __address: AddressFormat | None=None) -> int:
            """Similar to :meth:`socket.socket.sendmsg`, but async.

            Only available on platforms where :meth:`socket.socket.sendmsg` is
            available.

            """
            if __address is not None:
                __address = await self._resolve_address_nocp(__address, local=False)
            return await self._nonblocking_helper(_core.wait_writable, _stdlib_socket.socket.sendmsg, __buffers, __ancdata, __flags, __address)