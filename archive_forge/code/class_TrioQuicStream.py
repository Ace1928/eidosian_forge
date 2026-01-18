import socket
import ssl
import struct
import time
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import aioquic.quic.events  # type: ignore
import trio
import dns.exception
import dns.inet
from dns._asyncbackend import NullContext
from dns.quic._common import (
class TrioQuicStream(BaseQuicStream):

    def __init__(self, connection, stream_id):
        super().__init__(connection, stream_id)
        self._wake_up = trio.Condition()

    async def wait_for(self, amount):
        while True:
            if self._buffer.have(amount):
                return
            self._expecting = amount
            async with self._wake_up:
                await self._wake_up.wait()
            self._expecting = 0

    async def receive(self, timeout=None):
        if timeout is None:
            context = NullContext(None)
        else:
            context = trio.move_on_after(timeout)
        with context:
            await self.wait_for(2)
            size, = struct.unpack('!H', self._buffer.get(2))
            await self.wait_for(size)
            return self._buffer.get(size)
        raise dns.exception.Timeout

    async def send(self, datagram, is_end=False):
        data = self._encapsulate(datagram)
        await self._connection.write(self._stream_id, data, is_end)

    async def _add_input(self, data, is_end):
        if self._common_add_input(data, is_end):
            async with self._wake_up:
                self._wake_up.notify()

    async def close(self):
        self._close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        async with self._wake_up:
            self._wake_up.notify()
        return False