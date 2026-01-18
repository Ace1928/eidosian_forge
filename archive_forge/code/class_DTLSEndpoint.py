from __future__ import annotations
import contextlib
import enum
import errno
import hmac
import os
import struct
import warnings
import weakref
from itertools import count
from typing import (
from weakref import ReferenceType, WeakValueDictionary
import attrs
import trio
from ._util import NoPublicConstructor, final
@final
class DTLSEndpoint:
    """A DTLS endpoint.

    A single UDP socket can handle arbitrarily many DTLS connections simultaneously,
    acting as a client or server as needed. A `DTLSEndpoint` object holds a UDP socket
    and manages these connections, which are represented as `DTLSChannel` objects.

    Args:
      socket: (trio.socket.SocketType): A ``SOCK_DGRAM`` socket. If you want to accept
        incoming connections in server mode, then you should probably bind the socket to
        some known port.
      incoming_packets_buffer (int): Each `DTLSChannel` using this socket has its own
        buffer that holds incoming packets until you call `~DTLSChannel.receive` to read
        them. This lets you adjust the size of this buffer. `~DTLSChannel.statistics`
        lets you check if the buffer has overflowed.

    .. attribute:: socket
                   incoming_packets_buffer

       Both constructor arguments are also exposed as attributes, in case you need to
       access them later.

    """

    def __init__(self, socket: SocketType, *, incoming_packets_buffer: int=10) -> None:
        global SSL
        from OpenSSL import SSL
        self._initialized: bool = False
        if socket.type != trio.socket.SOCK_DGRAM:
            raise ValueError('DTLS requires a SOCK_DGRAM socket')
        self._initialized = True
        self.socket: SocketType = socket
        self.incoming_packets_buffer = incoming_packets_buffer
        self._token = trio.lowlevel.current_trio_token()
        self._streams: WeakValueDictionary[Any, DTLSChannel] = WeakValueDictionary()
        self._listening_context: SSL.Context | None = None
        self._listening_key: bytes | None = None
        self._incoming_connections_q = _Queue[DTLSChannel](float('inf'))
        self._send_lock = trio.Lock()
        self._closed = False
        self._receive_loop_spawned = False

    def _ensure_receive_loop(self) -> None:
        if not self._receive_loop_spawned:
            trio.lowlevel.spawn_system_task(dtls_receive_loop, weakref.ref(self), self.socket)
            self._receive_loop_spawned = True

    def __del__(self) -> None:
        if not self._initialized:
            return
        if not self._closed:
            with contextlib.suppress(RuntimeError):
                self._token.run_sync_soon(self.close)
            warnings.warn(f'unclosed DTLS endpoint {self!r}', ResourceWarning, source=self, stacklevel=1)

    def close(self) -> None:
        """Close this socket, and all associated DTLS connections.

        This object can also be used as a context manager.

        """
        self._closed = True
        self.socket.close()
        for stream in list(self._streams.values()):
            stream.close()
        self._incoming_connections_q.s.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        return self.close()

    def _check_closed(self) -> None:
        if self._closed:
            raise trio.ClosedResourceError

    async def serve(self, ssl_context: SSL.Context, async_fn: Callable[[DTLSChannel, Unpack[PosArgsT]], Awaitable[object]], *args: Unpack[PosArgsT], task_status: trio.TaskStatus[None]=trio.TASK_STATUS_IGNORED) -> None:
        """Listen for incoming connections, and spawn a handler for each using an
        internal nursery.

        Similar to `~trio.serve_tcp`, this function never returns until cancelled, or
        the `DTLSEndpoint` is closed and all handlers have exited.

        Usage commonly looks like::

            async def handler(dtls_channel):
                ...

            async with trio.open_nursery() as nursery:
                await nursery.start(dtls_endpoint.serve, ssl_context, handler)
                # ... do other things here ...

        The ``dtls_channel`` passed into the handler function has already performed the
        "cookie exchange" part of the DTLS handshake, so the peer address is
        trustworthy. But the actual cryptographic handshake doesn't happen until you
        start using it, giving you a chance for any last minute configuration, and the
        option to catch and handle handshake errors.

        Args:
          ssl_context (OpenSSL.SSL.Context): The PyOpenSSL context object to use for
            incoming connections.
          async_fn: The handler function that will be invoked for each incoming
            connection.
          *args: Additional arguments to pass to the handler function.

        """
        self._check_closed()
        if self._listening_context is not None:
            raise trio.BusyResourceError('another task is already listening')
        try:
            self.socket.getsockname()
        except OSError:
            raise RuntimeError('DTLS socket must be bound before it can serve') from None
        self._ensure_receive_loop()
        ssl_context.set_cookie_verify_callback(lambda *_: True)
        try:
            self._listening_context = ssl_context
            task_status.started()

            async def handler_wrapper(stream: DTLSChannel) -> None:
                with stream:
                    await async_fn(stream, *args)
            async with trio.open_nursery() as nursery:
                async for stream in self._incoming_connections_q.r:
                    nursery.start_soon(handler_wrapper, stream)
        finally:
            self._listening_context = None

    def connect(self, address: tuple[str, int], ssl_context: SSL.Context) -> DTLSChannel:
        """Initiate an outgoing DTLS connection.

        Notice that this is a synchronous method. That's because it doesn't actually
        initiate any I/O – it just sets up a `DTLSChannel` object. The actual handshake
        doesn't occur until you start using the `DTLSChannel`. This gives you a chance
        to do further configuration first, like setting MTU etc.

        Args:
          address: The address to connect to. Usually a (host, port) tuple, like
            ``("127.0.0.1", 12345)``.
          ssl_context (OpenSSL.SSL.Context): The PyOpenSSL context object to use for
            this connection.

        Returns:
          DTLSChannel

        """
        self._check_closed()
        channel = DTLSChannel._create(self, address, ssl_context)
        channel._ssl.set_connect_state()
        old_channel = self._streams.get(address)
        if old_channel is not None:
            old_channel._set_replaced()
        self._streams[address] = channel
        return channel