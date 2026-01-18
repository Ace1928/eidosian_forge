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
class RequestResponseCycle:

    def __init__(self, scope: HTTPScope, conn: h11.Connection, transport: asyncio.Transport, flow: FlowControl, logger: logging.Logger, access_logger: logging.Logger, access_log: bool, default_headers: list[tuple[bytes, bytes]], message_event: asyncio.Event, on_response: Callable[..., None]) -> None:
        self.scope = scope
        self.conn = conn
        self.transport = transport
        self.flow = flow
        self.logger = logger
        self.access_logger = access_logger
        self.access_log = access_log
        self.default_headers = default_headers
        self.message_event = message_event
        self.on_response = on_response
        self.disconnected = False
        self.keep_alive = True
        self.waiting_for_100_continue = conn.they_are_waiting_for_100_continue
        self.body = b''
        self.more_body = True
        self.response_started = False
        self.response_complete = False

    async def run_asgi(self, app: ASGI3Application) -> None:
        try:
            result = await app(self.scope, self.receive, self.send)
        except ClientDisconnected:
            pass
        except BaseException as exc:
            msg = 'Exception in ASGI application\n'
            self.logger.error(msg, exc_info=exc)
            if not self.response_started:
                await self.send_500_response()
            else:
                self.transport.close()
        else:
            if result is not None:
                msg = "ASGI callable should return None, but returned '%s'."
                self.logger.error(msg, result)
                self.transport.close()
            elif not self.response_started and (not self.disconnected):
                msg = 'ASGI callable returned without starting response.'
                self.logger.error(msg)
                await self.send_500_response()
            elif not self.response_complete and (not self.disconnected):
                msg = 'ASGI callable returned without completing response.'
                self.logger.error(msg)
                self.transport.close()
        finally:
            self.on_response = lambda: None

    async def send_500_response(self) -> None:
        response_start_event: HTTPResponseStartEvent = {'type': 'http.response.start', 'status': 500, 'headers': [(b'content-type', b'text/plain; charset=utf-8'), (b'connection', b'close')]}
        await self.send(response_start_event)
        response_body_event: HTTPResponseBodyEvent = {'type': 'http.response.body', 'body': b'Internal Server Error', 'more_body': False}
        await self.send(response_body_event)

    async def send(self, message: ASGISendEvent) -> None:
        message_type = message['type']
        if self.flow.write_paused and (not self.disconnected):
            await self.flow.drain()
        if self.disconnected:
            raise ClientDisconnected
        if not self.response_started:
            if message_type != 'http.response.start':
                msg = "Expected ASGI message 'http.response.start', but got '%s'."
                raise RuntimeError(msg % message_type)
            message = cast('HTTPResponseStartEvent', message)
            self.response_started = True
            self.waiting_for_100_continue = False
            status = message['status']
            headers = self.default_headers + list(message.get('headers', []))
            if CLOSE_HEADER in self.scope['headers'] and CLOSE_HEADER not in headers:
                headers = headers + [CLOSE_HEADER]
            if self.access_log:
                self.access_logger.info('%s - "%s %s HTTP/%s" %d', get_client_addr(self.scope), self.scope['method'], get_path_with_query_string(self.scope), self.scope['http_version'], status)
            reason = STATUS_PHRASES[status]
            response = h11.Response(status_code=status, headers=headers, reason=reason)
            output = self.conn.send(event=response)
            self.transport.write(output)
        elif not self.response_complete:
            if message_type != 'http.response.body':
                msg = "Expected ASGI message 'http.response.body', but got '%s'."
                raise RuntimeError(msg % message_type)
            message = cast('HTTPResponseBodyEvent', message)
            body = message.get('body', b'')
            more_body = message.get('more_body', False)
            data = b'' if self.scope['method'] == 'HEAD' else body
            output = self.conn.send(event=h11.Data(data=data))
            self.transport.write(output)
            if not more_body:
                self.response_complete = True
                self.message_event.set()
                output = self.conn.send(event=h11.EndOfMessage())
                self.transport.write(output)
        else:
            msg = "Unexpected ASGI message '%s' sent, after response already completed."
            raise RuntimeError(msg % message_type)
        if self.response_complete:
            if self.conn.our_state is h11.MUST_CLOSE or not self.keep_alive:
                self.conn.send(event=h11.ConnectionClosed())
                self.transport.close()
            self.on_response()

    async def receive(self) -> ASGIReceiveEvent:
        if self.waiting_for_100_continue and (not self.transport.is_closing()):
            headers: list[tuple[str, str]] = []
            event = h11.InformationalResponse(status_code=100, headers=headers, reason='Continue')
            output = self.conn.send(event=event)
            self.transport.write(output)
            self.waiting_for_100_continue = False
        if not self.disconnected and (not self.response_complete):
            self.flow.resume_reading()
            await self.message_event.wait()
            self.message_event.clear()
        if self.disconnected or self.response_complete:
            return {'type': 'http.disconnect'}
        message: HTTPRequestEvent = {'type': 'http.request', 'body': self.body, 'more_body': self.more_body}
        self.body = b''
        return message