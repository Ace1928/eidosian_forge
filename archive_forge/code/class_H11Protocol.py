from __future__ import annotations
import asyncio
import http
import logging
from typing import Any, Callable, Literal, cast
from urllib.parse import unquote
import h11
from h11._connection import DEFAULT_MAX_INCOMPLETE_EVENT_SIZE
from uvicorn._types import (
from uvicorn.config import Config
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.protocols.http.flow_control import (
from uvicorn.protocols.utils import (
from uvicorn.server import ServerState
class H11Protocol(asyncio.Protocol):

    def __init__(self, config: Config, server_state: ServerState, app_state: dict[str, Any], _loop: asyncio.AbstractEventLoop | None=None) -> None:
        if not config.loaded:
            config.load()
        self.config = config
        self.app = config.loaded_app
        self.loop = _loop or asyncio.get_event_loop()
        self.logger = logging.getLogger('uvicorn.error')
        self.access_logger = logging.getLogger('uvicorn.access')
        self.access_log = self.access_logger.hasHandlers()
        self.conn = h11.Connection(h11.SERVER, config.h11_max_incomplete_event_size if config.h11_max_incomplete_event_size is not None else DEFAULT_MAX_INCOMPLETE_EVENT_SIZE)
        self.ws_protocol_class = config.ws_protocol_class
        self.root_path = config.root_path
        self.limit_concurrency = config.limit_concurrency
        self.app_state = app_state
        self.timeout_keep_alive_task: asyncio.TimerHandle | None = None
        self.timeout_keep_alive = config.timeout_keep_alive
        self.server_state = server_state
        self.connections = server_state.connections
        self.tasks = server_state.tasks
        self.transport: asyncio.Transport = None
        self.flow: FlowControl = None
        self.server: tuple[str, int] | None = None
        self.client: tuple[str, int] | None = None
        self.scheme: Literal['http', 'https'] | None = None
        self.scope: HTTPScope = None
        self.headers: list[tuple[bytes, bytes]] = None
        self.cycle: RequestResponseCycle = None

    def connection_made(self, transport: asyncio.Transport) -> None:
        self.connections.add(self)
        self.transport = transport
        self.flow = FlowControl(transport)
        self.server = get_local_addr(transport)
        self.client = get_remote_addr(transport)
        self.scheme = 'https' if is_ssl(transport) else 'http'
        if self.logger.level <= TRACE_LOG_LEVEL:
            prefix = '%s:%d - ' % self.client if self.client else ''
            self.logger.log(TRACE_LOG_LEVEL, '%sHTTP connection made', prefix)

    def connection_lost(self, exc: Exception | None) -> None:
        self.connections.discard(self)
        if self.logger.level <= TRACE_LOG_LEVEL:
            prefix = '%s:%d - ' % self.client if self.client else ''
            self.logger.log(TRACE_LOG_LEVEL, '%sHTTP connection lost', prefix)
        if self.cycle and (not self.cycle.response_complete):
            self.cycle.disconnected = True
        if self.conn.our_state != h11.ERROR:
            event = h11.ConnectionClosed()
            try:
                self.conn.send(event)
            except h11.LocalProtocolError:
                pass
        if self.cycle is not None:
            self.cycle.message_event.set()
        if self.flow is not None:
            self.flow.resume_writing()
        if exc is None:
            self.transport.close()
            self._unset_keepalive_if_required()

    def eof_received(self) -> None:
        pass

    def _unset_keepalive_if_required(self) -> None:
        if self.timeout_keep_alive_task is not None:
            self.timeout_keep_alive_task.cancel()
            self.timeout_keep_alive_task = None

    def _get_upgrade(self) -> bytes | None:
        connection = []
        upgrade = None
        for name, value in self.headers:
            if name == b'connection':
                connection = [token.lower().strip() for token in value.split(b',')]
            if name == b'upgrade':
                upgrade = value.lower()
        if b'upgrade' in connection:
            return upgrade
        return None

    def _should_upgrade_to_ws(self) -> bool:
        if self.ws_protocol_class is None:
            if self.config.ws == 'auto':
                msg = 'Unsupported upgrade request.'
                self.logger.warning(msg)
                msg = 'No supported WebSocket library detected. Please use "pip install \'uvicorn[standard]\'", or install \'websockets\' or \'wsproto\' manually.'
                self.logger.warning(msg)
            return False
        return True

    def data_received(self, data: bytes) -> None:
        self._unset_keepalive_if_required()
        self.conn.receive_data(data)
        self.handle_events()

    def handle_events(self) -> None:
        while True:
            try:
                event = self.conn.next_event()
            except h11.RemoteProtocolError:
                msg = 'Invalid HTTP request received.'
                self.logger.warning(msg)
                self.send_400_response(msg)
                return
            if event is h11.NEED_DATA:
                break
            elif event is h11.PAUSED:
                self.flow.pause_reading()
                break
            elif isinstance(event, h11.Request):
                self.headers = [(key.lower(), value) for key, value in event.headers]
                raw_path, _, query_string = event.target.partition(b'?')
                path = unquote(raw_path.decode('ascii'))
                full_path = self.root_path + path
                full_raw_path = self.root_path.encode('ascii') + raw_path
                self.scope = {'type': 'http', 'asgi': {'version': self.config.asgi_version, 'spec_version': '2.4'}, 'http_version': event.http_version.decode('ascii'), 'server': self.server, 'client': self.client, 'scheme': self.scheme, 'method': event.method.decode('ascii'), 'root_path': self.root_path, 'path': full_path, 'raw_path': full_raw_path, 'query_string': query_string, 'headers': self.headers, 'state': self.app_state.copy()}
                upgrade = self._get_upgrade()
                if upgrade == b'websocket' and self._should_upgrade_to_ws():
                    self.handle_websocket_upgrade(event)
                    return
                if self.limit_concurrency is not None and (len(self.connections) >= self.limit_concurrency or len(self.tasks) >= self.limit_concurrency):
                    app = service_unavailable
                    message = 'Exceeded concurrency limit.'
                    self.logger.warning(message)
                else:
                    app = self.app
                self._unset_keepalive_if_required()
                self.cycle = RequestResponseCycle(scope=self.scope, conn=self.conn, transport=self.transport, flow=self.flow, logger=self.logger, access_logger=self.access_logger, access_log=self.access_log, default_headers=self.server_state.default_headers, message_event=asyncio.Event(), on_response=self.on_response_complete)
                task = self.loop.create_task(self.cycle.run_asgi(app))
                task.add_done_callback(self.tasks.discard)
                self.tasks.add(task)
            elif isinstance(event, h11.Data):
                if self.conn.our_state is h11.DONE:
                    continue
                self.cycle.body += event.data
                if len(self.cycle.body) > HIGH_WATER_LIMIT:
                    self.flow.pause_reading()
                self.cycle.message_event.set()
            elif isinstance(event, h11.EndOfMessage):
                if self.conn.our_state is h11.DONE:
                    self.transport.resume_reading()
                    self.conn.start_next_cycle()
                    continue
                self.cycle.more_body = False
                self.cycle.message_event.set()

    def handle_websocket_upgrade(self, event: h11.Request) -> None:
        if self.logger.level <= TRACE_LOG_LEVEL:
            prefix = '%s:%d - ' % self.client if self.client else ''
            self.logger.log(TRACE_LOG_LEVEL, '%sUpgrading to WebSocket', prefix)
        self.connections.discard(self)
        output = [event.method, b' ', event.target, b' HTTP/1.1\r\n']
        for name, value in self.headers:
            output += [name, b': ', value, b'\r\n']
        output.append(b'\r\n')
        protocol = self.ws_protocol_class(config=self.config, server_state=self.server_state, app_state=self.app_state)
        protocol.connection_made(self.transport)
        protocol.data_received(b''.join(output))
        self.transport.set_protocol(protocol)

    def send_400_response(self, msg: str) -> None:
        reason = STATUS_PHRASES[400]
        headers: list[tuple[bytes, bytes]] = [(b'content-type', b'text/plain; charset=utf-8'), (b'connection', b'close')]
        event = h11.Response(status_code=400, headers=headers, reason=reason)
        output = self.conn.send(event)
        self.transport.write(output)
        output = self.conn.send(event=h11.Data(data=msg.encode('ascii')))
        self.transport.write(output)
        output = self.conn.send(event=h11.EndOfMessage())
        self.transport.write(output)
        self.transport.close()

    def on_response_complete(self) -> None:
        self.server_state.total_requests += 1
        if self.transport.is_closing():
            return
        self._unset_keepalive_if_required()
        self.timeout_keep_alive_task = self.loop.call_later(self.timeout_keep_alive, self.timeout_keep_alive_handler)
        self.flow.resume_reading()
        if self.conn.our_state is h11.DONE and self.conn.their_state is h11.DONE:
            self.conn.start_next_cycle()
            self.handle_events()

    def shutdown(self) -> None:
        """
        Called by the server to commence a graceful shutdown.
        """
        if self.cycle is None or self.cycle.response_complete:
            event = h11.ConnectionClosed()
            self.conn.send(event)
            self.transport.close()
        else:
            self.cycle.keep_alive = False

    def pause_writing(self) -> None:
        """
        Called by the transport when the write buffer exceeds the high water mark.
        """
        self.flow.pause_writing()

    def resume_writing(self) -> None:
        """
        Called by the transport when the write buffer drops below the low water mark.
        """
        self.flow.resume_writing()

    def timeout_keep_alive_handler(self) -> None:
        """
        Called on a keep-alive connection if no new data is received after a short
        delay.
        """
        if not self.transport.is_closing():
            event = h11.ConnectionClosed()
            self.conn.send(event)
            self.transport.close()