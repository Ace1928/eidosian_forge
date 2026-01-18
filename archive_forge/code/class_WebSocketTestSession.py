from __future__ import annotations
import contextlib
import inspect
import io
import json
import math
import queue
import sys
import typing
import warnings
from concurrent.futures import Future
from functools import cached_property
from types import GeneratorType
from urllib.parse import unquote, urljoin
import anyio
import anyio.abc
import anyio.from_thread
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from anyio.streams.stapled import StapledObjectStream
from starlette._utils import is_async_callable
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect
class WebSocketTestSession:

    def __init__(self, app: ASGI3App, scope: Scope, portal_factory: _PortalFactoryType) -> None:
        self.app = app
        self.scope = scope
        self.accepted_subprotocol = None
        self.portal_factory = portal_factory
        self._receive_queue: queue.Queue[Message] = queue.Queue()
        self._send_queue: queue.Queue[Message | BaseException] = queue.Queue()
        self.extra_headers = None

    def __enter__(self) -> WebSocketTestSession:
        self.exit_stack = contextlib.ExitStack()
        self.portal = self.exit_stack.enter_context(self.portal_factory())
        try:
            _: Future[None] = self.portal.start_task_soon(self._run)
            self.send({'type': 'websocket.connect'})
            message = self.receive()
            self._raise_on_close(message)
        except Exception:
            self.exit_stack.close()
            raise
        self.accepted_subprotocol = message.get('subprotocol', None)
        self.extra_headers = message.get('headers', None)
        return self

    @cached_property
    def should_close(self) -> anyio.Event:
        return anyio.Event()

    async def _notify_close(self) -> None:
        self.should_close.set()

    def __exit__(self, *args: typing.Any) -> None:
        try:
            self.close(1000)
        finally:
            self.portal.start_task_soon(self._notify_close)
            self.exit_stack.close()
        while not self._send_queue.empty():
            message = self._send_queue.get()
            if isinstance(message, BaseException):
                raise message

    async def _run(self) -> None:
        """
        The sub-thread in which the websocket session runs.
        """

        async def run_app(tg: anyio.abc.TaskGroup) -> None:
            try:
                await self.app(self.scope, self._asgi_receive, self._asgi_send)
            except anyio.get_cancelled_exc_class():
                ...
            except BaseException as exc:
                self._send_queue.put(exc)
                raise
            finally:
                tg.cancel_scope.cancel()
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_app, tg)
            await self.should_close.wait()
            tg.cancel_scope.cancel()

    async def _asgi_receive(self) -> Message:
        while self._receive_queue.empty():
            await anyio.sleep(0)
        return self._receive_queue.get()

    async def _asgi_send(self, message: Message) -> None:
        self._send_queue.put(message)

    def _raise_on_close(self, message: Message) -> None:
        if message['type'] == 'websocket.close':
            raise WebSocketDisconnect(message.get('code', 1000), message.get('reason', ''))

    def send(self, message: Message) -> None:
        self._receive_queue.put(message)

    def send_text(self, data: str) -> None:
        self.send({'type': 'websocket.receive', 'text': data})

    def send_bytes(self, data: bytes) -> None:
        self.send({'type': 'websocket.receive', 'bytes': data})

    def send_json(self, data: typing.Any, mode: typing.Literal['text', 'binary']='text') -> None:
        text = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        if mode == 'text':
            self.send({'type': 'websocket.receive', 'text': text})
        else:
            self.send({'type': 'websocket.receive', 'bytes': text.encode('utf-8')})

    def close(self, code: int=1000, reason: str | None=None) -> None:
        self.send({'type': 'websocket.disconnect', 'code': code, 'reason': reason})

    def receive(self) -> Message:
        message = self._send_queue.get()
        if isinstance(message, BaseException):
            raise message
        return message

    def receive_text(self) -> str:
        message = self.receive()
        self._raise_on_close(message)
        return typing.cast(str, message['text'])

    def receive_bytes(self) -> bytes:
        message = self.receive()
        self._raise_on_close(message)
        return typing.cast(bytes, message['bytes'])

    def receive_json(self, mode: typing.Literal['text', 'binary']='text') -> typing.Any:
        message = self.receive()
        self._raise_on_close(message)
        if mode == 'text':
            text = message['text']
        else:
            text = message['bytes'].decode('utf-8')
        return json.loads(text)