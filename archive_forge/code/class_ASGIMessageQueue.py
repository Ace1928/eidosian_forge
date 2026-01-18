import asyncio
import inspect
import json
import logging
import pickle
import socket
from typing import Any, List, Optional, Type
import starlette
from fastapi.encoders import jsonable_encoder
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from uvicorn.config import Config
from uvicorn.lifespan.on import LifespanOn
from ray._private.pydantic_compat import IS_PYDANTIC_2
from ray.actor import ActorHandle
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import serve_encoders
from ray.serve.exceptions import RayServeException
class ASGIMessageQueue(Send):
    """Queue enables polling for received or sent messages.

    This class assumes a single consumer of the queue (concurrent calls to
    `get_messages_nowait` and `wait_for_message` is undefined behavior).
    """

    def __init__(self):
        self._message_queue = asyncio.Queue()
        self._new_message_event = asyncio.Event()
        self._closed = False

    def close(self):
        """Close the queue, rejecting new messages.

        Once the queue is closed, existing messages will be returned from
        `get_messages_nowait` and subsequent calls to `wait_for_message` will
        always return immediately.
        """
        self._closed = True
        self._new_message_event.set()

    async def __call__(self, message: Message):
        """Send a message, putting it on the queue.

        `RuntimeError` is raised if the queue has been closed using `.close()`.
        """
        if self._closed:
            raise RuntimeError('New messages cannot be sent after the queue is closed.')
        await self._message_queue.put(message)
        self._new_message_event.set()

    def get_messages_nowait(self) -> List[Message]:
        """Returns all messages that are currently available (non-blocking).

        At least one message will be present if `wait_for_message` had previously
        returned and a subsequent call to `wait_for_message` blocks until at
        least one new message is available.
        """
        messages = []
        while not self._message_queue.empty():
            messages.append(self._message_queue.get_nowait())
        self._new_message_event.clear()
        return messages

    async def wait_for_message(self):
        """Wait until at least one new message is available.

        If a message is available, this method will return immediately on each call
        until `get_messages_nowait` is called.

        After the queue is closed using `.close()`, this will always return
        immediately.
        """
        if not self._closed:
            await self._new_message_event.wait()