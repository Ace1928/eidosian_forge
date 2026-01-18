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
class TrioQuicConnection(AsyncQuicConnection):

    def __init__(self, connection, address, port, source, source_port, manager=None):
        super().__init__(connection, address, port, source, source_port, manager)
        self._socket = trio.socket.socket(self._af, socket.SOCK_DGRAM, 0)
        self._handshake_complete = trio.Event()
        self._run_done = trio.Event()
        self._worker_scope = None
        self._send_pending = False

    async def _worker(self):
        try:
            if self._source:
                await self._socket.bind(dns.inet.low_level_address_tuple(self._source, self._af))
            await self._socket.connect(self._peer)
            while not self._done:
                expiration, interval = self._get_timer_values(False)
                if self._send_pending:
                    interval = 0.0
                with trio.CancelScope(deadline=trio.current_time() + interval) as self._worker_scope:
                    datagram = await self._socket.recv(QUIC_MAX_DATAGRAM)
                    self._connection.receive_datagram(datagram, self._peer, time.time())
                self._worker_scope = None
                self._handle_timer(expiration)
                await self._handle_events()
                self._send_pending = False
                datagrams = self._connection.datagrams_to_send(time.time())
                for datagram, _ in datagrams:
                    await self._socket.send(datagram)
        finally:
            self._done = True
            self._handshake_complete.set()

    async def _handle_events(self):
        count = 0
        while True:
            event = self._connection.next_event()
            if event is None:
                return
            if isinstance(event, aioquic.quic.events.StreamDataReceived):
                stream = self._streams.get(event.stream_id)
                if stream:
                    await stream._add_input(event.data, event.end_stream)
            elif isinstance(event, aioquic.quic.events.HandshakeCompleted):
                self._handshake_complete.set()
            elif isinstance(event, aioquic.quic.events.ConnectionTerminated):
                self._done = True
                self._socket.close()
            elif isinstance(event, aioquic.quic.events.StreamReset):
                stream = self._streams.get(event.stream_id)
                if stream:
                    await stream._add_input(b'', True)
            count += 1
            if count > 10:
                count = 0
                await trio.sleep(0)

    async def write(self, stream, data, is_end=False):
        self._connection.send_stream_data(stream, data, is_end)
        self._send_pending = True
        if self._worker_scope is not None:
            self._worker_scope.cancel()

    async def run(self):
        if self._closed:
            return
        async with trio.open_nursery() as nursery:
            nursery.start_soon(self._worker)
        self._run_done.set()

    async def make_stream(self, timeout=None):
        if timeout is None:
            context = NullContext(None)
        else:
            context = trio.move_on_after(timeout)
        with context:
            await self._handshake_complete.wait()
            if self._done:
                raise UnexpectedEOF
            stream_id = self._connection.get_next_available_stream_id(False)
            stream = TrioQuicStream(self, stream_id)
            self._streams[stream_id] = stream
            return stream
        raise dns.exception.Timeout

    async def close(self):
        if not self._closed:
            self._manager.closed(self._peer[0], self._peer[1])
            self._closed = True
            self._connection.close()
            self._send_pending = True
            if self._worker_scope is not None:
                self._worker_scope.cancel()
            await self._run_done.wait()