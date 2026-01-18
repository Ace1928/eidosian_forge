from __future__ import annotations
import asyncio
import http
import logging
import re
import urllib
from asyncio.events import TimerHandle
from collections import deque
from typing import Any, Callable, Literal, cast
import httptools
from uvicorn._types import (
from uvicorn.config import Config
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.protocols.http.flow_control import (
from uvicorn.protocols.utils import (
from uvicorn.server import ServerState
class HttpToolsProtocol(asyncio.Protocol):

    def __init__(self, config: Config, server_state: ServerState, app_state: dict[str, Any], _loop: asyncio.AbstractEventLoop | None=None) -> None:
        if not config.loaded:
            config.load()
        self.config = config
        self.app = config.loaded_app
        self.loop = _loop or asyncio.get_event_loop()
        self.logger = logging.getLogger('uvicorn.error')
        self.access_logger = logging.getLogger('uvicorn.access')
        self.access_log = self.access_logger.hasHandlers()
        self.parser = httptools.HttpRequestParser(self)
        self.ws_protocol_class = config.ws_protocol_class
        self.root_path = config.root_path
        self.limit_concurrency = config.limit_concurrency
        self.app_state = app_state
        self.timeout_keep_alive_task: TimerHandle | None = None
        self.timeout_keep_alive = config.timeout_keep_alive
        self.server_state = server_state
        self.connections = server_state.connections
        self.tasks = server_state.tasks
        self.transport: asyncio.Transport = None
        self.flow: FlowControl = None
        self.server: tuple[str, int] | None = None
        self.client: tuple[str, int] | None = None
        self.scheme: Literal['http', 'https'] | None = None
        self.pipeline: deque[tuple[RequestResponseCycle, ASGI3Application]] = deque()
        self.scope: HTTPScope = None
        self.headers: list[tuple[bytes, bytes]] = None
        self.expect_100_continue = False
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
        if self.cycle is not None:
            self.cycle.message_event.set()
        if self.flow is not None:
            self.flow.resume_writing()
        if exc is None:
            self.transport.close()
            self._unset_keepalive_if_required()
        self.parser = None

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

    def _should_upgrade_to_ws(self, upgrade: bytes | None) -> bool:
        if upgrade == b'websocket' and self.ws_protocol_class is not None:
            return True
        if self.config.ws == 'auto':
            msg = 'Unsupported upgrade request.'
            self.logger.warning(msg)
            msg = 'No supported WebSocket library detected. Please use "pip install \'uvicorn[standard]\'", or install \'websockets\' or \'wsproto\' manually.'
            self.logger.warning(msg)
        return False

    def _should_upgrade(self) -> bool:
        upgrade = self._get_upgrade()
        return self._should_upgrade_to_ws(upgrade)

    def data_received(self, data: bytes) -> None:
        self._unset_keepalive_if_required()
        try:
            self.parser.feed_data(data)
        except httptools.HttpParserError:
            msg = 'Invalid HTTP request received.'
            self.logger.warning(msg)
            self.send_400_response(msg)
            return
        except httptools.HttpParserUpgrade:
            upgrade = self._get_upgrade()
            if self._should_upgrade_to_ws(upgrade):
                self.handle_websocket_upgrade()

    def handle_websocket_upgrade(self) -> None:
        if self.logger.level <= TRACE_LOG_LEVEL:
            prefix = '%s:%d - ' % self.client if self.client else ''
            self.logger.log(TRACE_LOG_LEVEL, '%sUpgrading to WebSocket', prefix)
        self.connections.discard(self)
        method = self.scope['method'].encode()
        output = [method, b' ', self.url, b' HTTP/1.1\r\n']
        for name, value in self.scope['headers']:
            output += [name, b': ', value, b'\r\n']
        output.append(b'\r\n')
        protocol = self.ws_protocol_class(config=self.config, server_state=self.server_state, app_state=self.app_state)
        protocol.connection_made(self.transport)
        protocol.data_received(b''.join(output))
        self.transport.set_protocol(protocol)

    def send_400_response(self, msg: str) -> None:
        content = [STATUS_LINE[400]]
        for name, value in self.server_state.default_headers:
            content.extend([name, b': ', value, b'\r\n'])
        content.extend([b'content-type: text/plain; charset=utf-8\r\n', b'content-length: ' + str(len(msg)).encode('ascii') + b'\r\n', b'connection: close\r\n', b'\r\n', msg.encode('ascii')])
        self.transport.write(b''.join(content))
        self.transport.close()

    def on_message_begin(self) -> None:
        self.url = b''
        self.expect_100_continue = False
        self.headers = []
        self.scope = {'type': 'http', 'asgi': {'version': self.config.asgi_version, 'spec_version': '2.4'}, 'http_version': '1.1', 'server': self.server, 'client': self.client, 'scheme': self.scheme, 'root_path': self.root_path, 'headers': self.headers, 'state': self.app_state.copy()}

    def on_url(self, url: bytes) -> None:
        self.url += url

    def on_header(self, name: bytes, value: bytes) -> None:
        name = name.lower()
        if name == b'expect' and value.lower() == b'100-continue':
            self.expect_100_continue = True
        self.headers.append((name, value))

    def on_headers_complete(self) -> None:
        http_version = self.parser.get_http_version()
        method = self.parser.get_method()
        self.scope['method'] = method.decode('ascii')
        if http_version != '1.1':
            self.scope['http_version'] = http_version
        if self.parser.should_upgrade() and self._should_upgrade():
            return
        parsed_url = httptools.parse_url(self.url)
        raw_path = parsed_url.path
        path = raw_path.decode('ascii')
        if '%' in path:
            path = urllib.parse.unquote(path)
        full_path = self.root_path + path
        full_raw_path = self.root_path.encode('ascii') + raw_path
        self.scope['path'] = full_path
        self.scope['raw_path'] = full_raw_path
        self.scope['query_string'] = parsed_url.query or b''
        if self.limit_concurrency is not None and (len(self.connections) >= self.limit_concurrency or len(self.tasks) >= self.limit_concurrency):
            app = service_unavailable
            message = 'Exceeded concurrency limit.'
            self.logger.warning(message)
        else:
            app = self.app
        existing_cycle = self.cycle
        self.cycle = RequestResponseCycle(scope=self.scope, transport=self.transport, flow=self.flow, logger=self.logger, access_logger=self.access_logger, access_log=self.access_log, default_headers=self.server_state.default_headers, message_event=asyncio.Event(), expect_100_continue=self.expect_100_continue, keep_alive=http_version != '1.0', on_response=self.on_response_complete)
        if existing_cycle is None or existing_cycle.response_complete:
            task = self.loop.create_task(self.cycle.run_asgi(app))
            task.add_done_callback(self.tasks.discard)
            self.tasks.add(task)
        else:
            self.flow.pause_reading()
            self.pipeline.appendleft((self.cycle, app))

    def on_body(self, body: bytes) -> None:
        if self.parser.should_upgrade() and self._should_upgrade() or self.cycle.response_complete:
            return
        self.cycle.body += body
        if len(self.cycle.body) > HIGH_WATER_LIMIT:
            self.flow.pause_reading()
        self.cycle.message_event.set()

    def on_message_complete(self) -> None:
        if self.parser.should_upgrade() and self._should_upgrade() or self.cycle.response_complete:
            return
        self.cycle.more_body = False
        self.cycle.message_event.set()

    def on_response_complete(self) -> None:
        self.server_state.total_requests += 1
        if self.transport.is_closing():
            return
        self._unset_keepalive_if_required()
        self.flow.resume_reading()
        if self.pipeline:
            cycle, app = self.pipeline.pop()
            task = self.loop.create_task(cycle.run_asgi(app))
            task.add_done_callback(self.tasks.discard)
            self.tasks.add(task)
        else:
            self.timeout_keep_alive_task = self.loop.call_later(self.timeout_keep_alive, self.timeout_keep_alive_handler)

    def shutdown(self) -> None:
        """
        Called by the server to commence a graceful shutdown.
        """
        if self.cycle is None or self.cycle.response_complete:
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
            self.transport.close()