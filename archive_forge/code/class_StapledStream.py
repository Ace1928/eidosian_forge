from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar
import attrs
import trio
from trio._util import final
from .abc import AsyncResource, HalfCloseableStream, ReceiveStream, SendStream
@final
@attrs.define(eq=False, hash=False, slots=False)
class StapledStream(HalfCloseableStream, Generic[SendStreamT, ReceiveStreamT]):
    """This class `staples <https://en.wikipedia.org/wiki/Staple_(fastener)>`__
    together two unidirectional streams to make single bidirectional stream.

    Args:
      send_stream (~trio.abc.SendStream): The stream to use for sending.
      receive_stream (~trio.abc.ReceiveStream): The stream to use for
          receiving.

    Example:

       A silly way to make a stream that echoes back whatever you write to
       it::

          left, right = trio.testing.memory_stream_pair()
          echo_stream = StapledStream(SocketStream(left), SocketStream(right))
          await echo_stream.send_all(b"x")
          assert await echo_stream.receive_some() == b"x"

    :class:`StapledStream` objects implement the methods in the
    :class:`~trio.abc.HalfCloseableStream` interface. They also have two
    additional public attributes:

    .. attribute:: send_stream

       The underlying :class:`~trio.abc.SendStream`. :meth:`send_all` and
       :meth:`wait_send_all_might_not_block` are delegated to this object.

    .. attribute:: receive_stream

       The underlying :class:`~trio.abc.ReceiveStream`. :meth:`receive_some`
       is delegated to this object.

    """
    send_stream: SendStreamT
    receive_stream: ReceiveStreamT

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        """Calls ``self.send_stream.send_all``."""
        return await self.send_stream.send_all(data)

    async def wait_send_all_might_not_block(self) -> None:
        """Calls ``self.send_stream.wait_send_all_might_not_block``."""
        return await self.send_stream.wait_send_all_might_not_block()

    async def send_eof(self) -> None:
        """Shuts down the send side of the stream.

        If :meth:`self.send_stream.send_eof() <trio.abc.HalfCloseableStream.send_eof>` exists,
        then this calls it. Otherwise, this calls
        :meth:`self.send_stream.aclose() <trio.abc.AsyncResource.aclose>`.
        """
        stream = self.send_stream
        if _is_halfclosable(stream):
            return await stream.send_eof()
        else:
            return await stream.aclose()

    async def receive_some(self, max_bytes: int | None=None) -> bytes:
        """Calls ``self.receive_stream.receive_some``."""
        return await self.receive_stream.receive_some(max_bytes)

    async def aclose(self) -> None:
        """Calls ``aclose`` on both underlying streams."""
        try:
            await self.send_stream.aclose()
        finally:
            await self.receive_stream.aclose()