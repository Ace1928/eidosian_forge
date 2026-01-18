import contextvars
import importlib
import itertools
import json
import logging
import pathlib
import typing
from collections import defaultdict
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
import trio
from trio_websocket import ConnectionClosed as WsConnectionClosed
from trio_websocket import connect_websocket_url
class CdpBase:

    def __init__(self, ws, session_id, target_id):
        self.ws = ws
        self.session_id = session_id
        self.target_id = target_id
        self.channels = defaultdict(set)
        self.id_iter = itertools.count()
        self.inflight_cmd = {}
        self.inflight_result = {}

    async def execute(self, cmd: typing.Generator[dict, T, typing.Any]) -> T:
        """Execute a command on the server and wait for the result.

        :param cmd: any CDP command
        :returns: a CDP result
        """
        cmd_id = next(self.id_iter)
        cmd_event = trio.Event()
        self.inflight_cmd[cmd_id] = (cmd, cmd_event)
        request = next(cmd)
        request['id'] = cmd_id
        if self.session_id:
            request['sessionId'] = self.session_id
        request_str = json.dumps(request)
        try:
            await self.ws.send_message(request_str)
        except WsConnectionClosed as wcc:
            raise CdpConnectionClosed(wcc.reason) from None
        await cmd_event.wait()
        response = self.inflight_result.pop(cmd_id)
        if isinstance(response, Exception):
            raise response
        return response

    def listen(self, *event_types, buffer_size=10):
        """Return an async iterator that iterates over events matching the
        indicated types."""
        sender, receiver = trio.open_memory_channel(buffer_size)
        for event_type in event_types:
            self.channels[event_type].add(sender)
        return receiver

    @asynccontextmanager
    async def wait_for(self, event_type: typing.Type[T], buffer_size=10) -> typing.AsyncGenerator[CmEventProxy, None]:
        """Wait for an event of the given type and return it.

        This is an async context manager, so you should open it inside
        an async with block. The block will not exit until the indicated
        event is received.
        """
        sender, receiver = trio.open_memory_channel(buffer_size)
        self.channels[event_type].add(sender)
        proxy = CmEventProxy()
        yield proxy
        async with receiver:
            event = await receiver.receive()
        proxy.value = event

    def _handle_data(self, data):
        """Handle incoming WebSocket data.

        :param dict data: a JSON dictionary
        """
        if 'id' in data:
            self._handle_cmd_response(data)
        else:
            self._handle_event(data)

    def _handle_cmd_response(self, data):
        """Handle a response to a command. This will set an event flag that
        will return control to the task that called the command.

        :param dict data: response as a JSON dictionary
        """
        cmd_id = data['id']
        try:
            cmd, event = self.inflight_cmd.pop(cmd_id)
        except KeyError:
            logger.warning('Got a message with a command ID that does not exist: %s', data)
            return
        if 'error' in data:
            self.inflight_result[cmd_id] = BrowserError(data['error'])
        else:
            try:
                _ = cmd.send(data['result'])
                raise InternalError("The command's generator function did not exit when expected!")
            except StopIteration as exit:
                return_ = exit.value
            self.inflight_result[cmd_id] = return_
        event.set()

    def _handle_event(self, data):
        """Handle an event.

        :param dict data: event as a JSON dictionary
        """
        global devtools
        event = devtools.util.parse_json_event(data)
        logger.debug('Received event: %s', event)
        to_remove = set()
        for sender in self.channels[type(event)]:
            try:
                sender.send_nowait(event)
            except trio.WouldBlock:
                logger.error('Unable to send event "%r" due to full channel %s', event, sender)
            except trio.BrokenResourceError:
                to_remove.add(sender)
        if to_remove:
            self.channels[type(event)] -= to_remove