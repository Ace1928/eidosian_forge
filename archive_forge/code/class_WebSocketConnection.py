import sys
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import partial
from ipaddress import ip_address
import itertools
import logging
import random
import ssl
import struct
import urllib.parse
from typing import List, Optional, Union
import trio
import trio.abc
from wsproto import ConnectionType, WSConnection
from wsproto.connection import ConnectionState
import wsproto.frame_protocol as wsframeproto
from wsproto.events import (
import wsproto.utilities
class WebSocketConnection(trio.abc.AsyncResource):
    """ A WebSocket connection. """
    CONNECTION_ID = itertools.count()

    def __init__(self, stream, ws_connection, *, host=None, path=None, client_subprotocols=None, client_extra_headers=None, message_queue_size=MESSAGE_QUEUE_SIZE, max_message_size=MAX_MESSAGE_SIZE):
        """
        Constructor.

        Generally speaking, users are discouraged from directly instantiating a
        ``WebSocketConnection`` and should instead use one of the convenience
        functions in this module, e.g. ``open_websocket()`` or
        ``serve_websocket()``. This class has some tricky internal logic and
        timing that depends on whether it is an instance of a client connection
        or a server connection. The convenience functions handle this complexity
        for you.

        :param SocketStream stream:
        :param ws_connection wsproto.WSConnection:
        :param str host: The hostname to send in the HTTP request headers. Only
            used for client connections.
        :param str path: The URL path for this connection.
        :param list client_subprotocols: A list of desired subprotocols. Only
            used for client connections.
        :param list[tuple[bytes,bytes]] client_extra_headers: Extra headers to
            send with the connection request. Only used for client connections.
        :param int message_queue_size: The maximum number of messages that will be
            buffered in the library's internal message queue.
        :param int max_message_size: The maximum message size as measured by
            ``len()``. If a message is received that is larger than this size,
            then the connection is closed with code 1009 (Message Too Big).
        """
        self._close_reason: Optional[CloseReason] = None
        self._id = next(self.__class__.CONNECTION_ID)
        self._stream = stream
        self._stream_lock = trio.StrictFIFOLock()
        self._wsproto = ws_connection
        self._message_size = 0
        self._message_parts: List[Union[bytes, str]] = []
        self._max_message_size = max_message_size
        self._reader_running = True
        if ws_connection.client:
            self._initial_request: Optional[Request] = Request(host=host, target=path, subprotocols=client_subprotocols, extra_headers=client_extra_headers or [])
        else:
            self._initial_request = None
        self._path = path
        self._subprotocol: Optional[str] = None
        self._handshake_headers = tuple()
        self._reject_status = 0
        self._reject_headers = tuple()
        self._reject_body = b''
        self._send_channel, self._recv_channel = trio.open_memory_channel(message_queue_size)
        self._pings = OrderedDict()
        self._connection_proposal = Future()
        self._open_handshake = trio.Event()
        self._close_handshake = trio.Event()
        self._for_testing_peer_closed_connection = trio.Event()

    @property
    def closed(self):
        """
        (Read-only) The reason why the connection was or is being closed,
        else ``None``.

        :rtype: Optional[CloseReason]
        """
        return self._close_reason

    @property
    def is_client(self):
        """ (Read-only) Is this a client instance? """
        return self._wsproto.client

    @property
    def is_server(self):
        """ (Read-only) Is this a server instance? """
        return not self._wsproto.client

    @property
    def local(self):
        """
        The local endpoint of the connection.

        :rtype: Endpoint or str
        """
        return _get_stream_endpoint(self._stream, local=True)

    @property
    def remote(self):
        """
        The remote endpoint of the connection.

        :rtype: Endpoint or str
        """
        return _get_stream_endpoint(self._stream, local=False)

    @property
    def path(self):
        """
        The requested URL path. For clients, this is set when the connection is
        instantiated. For servers, it is set after the handshake completes.

        :rtype: str
        """
        return self._path

    @property
    def subprotocol(self):
        """
        (Read-only) The negotiated subprotocol, or ``None`` if there is no
        subprotocol.

        This is only valid after the opening handshake is complete.

        :rtype: str or None
        """
        return self._subprotocol

    @property
    def handshake_headers(self):
        """
        The HTTP headers that were sent by the remote during the handshake,
        stored as 2-tuples containing key/value pairs. Header keys are always
        lower case.

        :rtype: tuple[tuple[str,str]]
        """
        return self._handshake_headers

    async def aclose(self, code=1000, reason=None):
        """
        Close the WebSocket connection.

        This sends a closing frame and suspends until the connection is closed.
        After calling this method, any further I/O on this WebSocket (such as
        ``get_message()`` or ``send_message()``) will raise
        ``ConnectionClosed``.

        This method is idempotent: it may be called multiple times on the same
        connection without any errors.

        :param int code: A 4-digit code number indicating the type of closure.
        :param str reason: An optional string describing the closure.
        """
        with _preserve_current_exception():
            await self._aclose(code, reason)

    async def _aclose(self, code, reason):
        if self._close_reason:
            return
        try:
            if self._wsproto.state == ConnectionState.OPEN:
                self._close_reason = CloseReason(1000, None)
                await self._send(CloseConnection(code=code, reason=reason))
            elif self._wsproto.state in (ConnectionState.CONNECTING, ConnectionState.REJECTING):
                self._close_handshake.set()
            await self._recv_channel.aclose()
            await self._close_handshake.wait()
        except ConnectionClosed:
            pass
        finally:
            await self._close_stream()

    async def get_message(self):
        """
        Receive the next WebSocket message.

        If no message is available immediately, then this function blocks until
        a message is ready.

        If the remote endpoint closes the connection, then the caller can still
        get messages sent prior to closing. Once all pending messages have been
        retrieved, additional calls to this method will raise
        ``ConnectionClosed``. If the local endpoint closes the connection, then
        pending messages are discarded and calls to this method will immediately
        raise ``ConnectionClosed``.

        :rtype: str or bytes
        :raises ConnectionClosed: if the connection is closed.
        """
        try:
            message = await self._recv_channel.receive()
        except (trio.ClosedResourceError, trio.EndOfChannel):
            raise ConnectionClosed(self._close_reason) from None
        return message

    async def ping(self, payload=None):
        """
        Send WebSocket ping to remote endpoint and wait for a correspoding pong.

        Each in-flight ping must include a unique payload. This function sends
        the ping and then waits for a corresponding pong from the remote
        endpoint.

        *Note: If the remote endpoint recieves multiple pings, it is allowed to
        send a single pong. Therefore, the order of calls to ``ping()`` is
        tracked, and a pong will wake up its corresponding ping as well as all
        previous in-flight pings.*

        :param payload: The payload to send. If ``None`` then a random 32-bit
            payload is created.
        :type payload: bytes or None
        :raises ConnectionClosed: if connection is closed.
        :raises ValueError: if ``payload`` is identical to another in-flight
            ping.
        """
        if self._close_reason:
            raise ConnectionClosed(self._close_reason)
        if payload in self._pings:
            raise ValueError(f'Payload value {payload} is already in flight.')
        if payload is None:
            payload = struct.pack('!I', random.getrandbits(32))
        event = trio.Event()
        self._pings[payload] = event
        await self._send(Ping(payload=payload))
        await event.wait()

    async def pong(self, payload=None):
        """
        Send an unsolicted pong.

        :param payload: The pong's payload. If ``None``, then no payload is
            sent.
        :type payload: bytes or None
        :raises ConnectionClosed: if connection is closed
        """
        if self._close_reason:
            raise ConnectionClosed(self._close_reason)
        await self._send(Pong(payload=payload))

    async def send_message(self, message):
        """
        Send a WebSocket message.

        :param message: The message to send.
        :type message: str or bytes
        :raises ConnectionClosed: if connection is closed, or being closed
        """
        if self._close_reason:
            raise ConnectionClosed(self._close_reason)
        if isinstance(message, str):
            event = TextMessage(data=message)
        elif isinstance(message, bytes):
            event = BytesMessage(data=message)
        else:
            raise ValueError('message must be str or bytes')
        await self._send(event)

    def __str__(self):
        """ Connection ID and type. """
        type_ = 'client' if self.is_client else 'server'
        return f'{type_}-{self._id}'

    async def _accept(self, request, subprotocol, extra_headers):
        """
        Accept the handshake.

        This method is only applicable to server-side connections.

        :param wsproto.events.Request request:
        :param subprotocol:
        :type subprotocol: str or None
        :param list[tuple[bytes,bytes]] extra_headers: A list of 2-tuples
            containing key/value pairs to send as HTTP headers.
        """
        self._subprotocol = subprotocol
        self._path = request.target
        await self._send(AcceptConnection(subprotocol=self._subprotocol, extra_headers=extra_headers))
        self._open_handshake.set()

    async def _reject(self, status_code, headers, body):
        """
        Reject the handshake.

        :param int status_code: The 3 digit HTTP status code. In order to be
            RFC-compliant, this must not be 101, and should be an appropriate
            code in the range 300-599.
        :param list[tuple[bytes,bytes]] headers: A list of 2-tuples containing
            key/value pairs to send as HTTP headers.
        :param bytes body: An optional response body.
        """
        if body:
            headers.append(('Content-length', str(len(body)).encode('ascii')))
        reject_conn = RejectConnection(status_code=status_code, headers=headers, has_body=bool(body))
        await self._send(reject_conn)
        if body:
            reject_body = RejectData(data=body)
            await self._send(reject_body)
        self._close_reason = CloseReason(1006, 'Rejected WebSocket handshake')
        self._close_handshake.set()

    async def _abort_web_socket(self):
        """
        If a stream is closed outside of this class, e.g. due to network
        conditions or because some other code closed our stream object, then we
        cannot perform the close handshake. We just need to clean up internal
        state.
        """
        close_reason = wsframeproto.CloseReason.ABNORMAL_CLOSURE
        if self._wsproto.state == ConnectionState.OPEN:
            self._wsproto.send(CloseConnection(code=close_reason.value))
        if self._close_reason is None:
            await self._close_web_socket(close_reason)
        self._reader_running = False
        self._close_handshake.set()

    async def _close_stream(self):
        """ Close the TCP connection. """
        self._reader_running = False
        try:
            with _preserve_current_exception():
                await self._stream.aclose()
        except trio.BrokenResourceError:
            pass

    async def _close_web_socket(self, code, reason=None):
        """
        Mark the WebSocket as closed. Close the message channel so that if any
        tasks are suspended in get_message(), they will wake up with a
        ConnectionClosed exception.
        """
        self._close_reason = CloseReason(code, reason)
        exc = ConnectionClosed(self._close_reason)
        logger.debug('%s websocket closed %r', self, exc)
        await self._send_channel.aclose()

    async def _get_request(self):
        """
        Return a proposal for a WebSocket handshake.

        This method can only be called on server connections and it may only be
        called one time.

        :rtype: WebSocketRequest
        """
        if not self.is_server:
            raise RuntimeError('This method is only valid for server connections.')
        if self._connection_proposal is None:
            raise RuntimeError('No proposal available. Did you call this method multiple times or at the wrong time?')
        proposal = await self._connection_proposal.wait_value()
        self._connection_proposal = None
        return proposal

    async def _handle_request_event(self, event):
        """
        Handle a connection request.

        This method is async even though it never awaits, because the event
        dispatch requires an async function.

        :param event:
        """
        proposal = WebSocketRequest(self, event)
        self._connection_proposal.set_value(proposal)

    async def _handle_accept_connection_event(self, event):
        """
        Handle an AcceptConnection event.

        :param wsproto.eventsAcceptConnection event:
        """
        self._subprotocol = event.subprotocol
        self._handshake_headers = tuple(event.extra_headers)
        self._open_handshake.set()

    async def _handle_reject_connection_event(self, event):
        """
        Handle a RejectConnection event.

        :param event:
        """
        self._reject_status = event.status_code
        self._reject_headers = tuple(event.headers)
        if not event.has_body:
            raise ConnectionRejected(self._reject_status, self._reject_headers, body=None)

    async def _handle_reject_data_event(self, event):
        """
        Handle a RejectData event.

        :param event:
        """
        self._reject_body += event.data
        if event.body_finished:
            raise ConnectionRejected(self._reject_status, self._reject_headers, body=self._reject_body)

    async def _handle_close_connection_event(self, event):
        """
        Handle a close event.

        :param wsproto.events.CloseConnection event:
        """
        if self._wsproto.state == ConnectionState.REMOTE_CLOSING:
            self._close_reason = CloseReason(event.code, event.reason or None)
            self._for_testing_peer_closed_connection.set()
            await self._send(event.response())
        await self._close_web_socket(event.code, event.reason or None)
        self._close_handshake.set()
        if self.is_server:
            await self._close_stream()

    async def _handle_message_event(self, event):
        """
        Handle a message event.

        :param event:
        :type event: wsproto.events.BytesMessage or wsproto.events.TextMessage
        """
        self._message_size += len(event.data)
        self._message_parts.append(event.data)
        if self._message_size > self._max_message_size:
            err = f'Exceeded maximum message size: {self._max_message_size} bytes'
            self._message_size = 0
            self._message_parts = []
            self._close_reason = CloseReason(1009, err)
            await self._send(CloseConnection(code=1009, reason=err))
            await self._recv_channel.aclose()
            self._reader_running = False
        elif event.message_finished:
            msg = (b'' if isinstance(event, BytesMessage) else '').join(self._message_parts)
            self._message_size = 0
            self._message_parts = []
            try:
                await self._send_channel.send(msg)
            except (trio.ClosedResourceError, trio.BrokenResourceError):
                pass

    async def _handle_ping_event(self, event):
        """
        Handle a PingReceived event.

        Wsproto queues a pong frame automatically, so this handler just needs to
        send it.

        :param wsproto.events.Ping event:
        """
        logger.debug('%s ping %r', self, event.payload)
        await self._send(event.response())

    async def _handle_pong_event(self, event):
        """
        Handle a PongReceived event.

        When a pong is received, check if we have any ping requests waiting for
        this pong response. If the remote endpoint skipped any earlier pings,
        then we wake up those skipped pings, too.

        This function is async even though it never awaits, because the other
        event handlers are async, too, and event dispatch would be more
        complicated if some handlers were sync.

        :param event:
        """
        payload = bytes(event.payload)
        try:
            event = self._pings[payload]
        except KeyError:
            return
        while self._pings:
            key, event = self._pings.popitem(0)
            skipped = ' [skipped] ' if payload != key else ' '
            logger.debug('%s pong%s%r', self, skipped, key)
            event.set()
            if payload == key:
                break

    async def _reader_task(self):
        """ A background task that reads network data and generates events. """
        handlers = {AcceptConnection: self._handle_accept_connection_event, BytesMessage: self._handle_message_event, CloseConnection: self._handle_close_connection_event, Ping: self._handle_ping_event, Pong: self._handle_pong_event, RejectConnection: self._handle_reject_connection_event, RejectData: self._handle_reject_data_event, Request: self._handle_request_event, TextMessage: self._handle_message_event}
        if self._initial_request:
            try:
                await self._send(self._initial_request)
            except ConnectionClosed:
                self._reader_running = False
        async with self._send_channel:
            while self._reader_running:
                for event in self._wsproto.events():
                    event_type = type(event)
                    try:
                        handler = handlers[event_type]
                        logger.debug('%s received event: %s', self, event_type)
                        await handler(event)
                    except KeyError:
                        logger.warning('%s received unknown event type: "%s"', self, event_type)
                    except ConnectionClosed:
                        self._reader_running = False
                        break
                try:
                    data = await self._stream.receive_some(RECEIVE_BYTES)
                except (trio.BrokenResourceError, trio.ClosedResourceError):
                    await self._abort_web_socket()
                    break
                if len(data) == 0:
                    logger.debug('%s received zero bytes (connection closed)', self)
                    if self._wsproto.state != ConnectionState.CLOSED:
                        await self._abort_web_socket()
                    break
                logger.debug('%s received %d bytes', self, len(data))
                if self._wsproto.state != ConnectionState.CLOSED:
                    try:
                        self._wsproto.receive_data(data)
                    except wsproto.utilities.RemoteProtocolError as err:
                        logger.debug('%s remote protocol error: %s', self, err)
                        if err.event_hint:
                            await self._send(err.event_hint)
                        await self._close_stream()
        logger.debug('%s reader task finished', self)

    async def _send(self, event):
        """
        Send an event to the remote WebSocket.

        The reader task and one or more writers might try to send messages at
        the same time, so this method uses an internal lock to serialize
        requests to send data.

        :param wsproto.events.Event event:
        """
        data = self._wsproto.send(event)
        async with self._stream_lock:
            logger.debug('%s sending %d bytes', self, len(data))
            try:
                await self._stream.send_all(data)
            except (trio.BrokenResourceError, trio.ClosedResourceError):
                await self._abort_web_socket()
                raise ConnectionClosed(self._close_reason) from None