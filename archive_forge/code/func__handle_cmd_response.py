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