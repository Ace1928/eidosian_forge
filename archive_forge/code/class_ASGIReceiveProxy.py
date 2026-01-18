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
class ASGIReceiveProxy:
    """Proxies ASGI receive from an actor.

    The provided actor handle is expected to implement a single method:
    `receive_asgi_messages`. It will be called repeatedly until a disconnect message
    is received.
    """

    def __init__(self, request_id: str, actor_handle: ActorHandle):
        self._queue = asyncio.Queue()
        self._request_id = request_id
        self._actor_handle = actor_handle
        self._disconnect_message = None

    async def fetch_until_disconnect(self):
        """Fetch messages repeatedly until a disconnect message is received.

        If a disconnect message is received, this function exits and returns it.

        If an exception occurs, it will be raised on the next __call__ and no more
        messages will be received.
        """
        while True:
            try:
                pickled_messages = await self._actor_handle.receive_asgi_messages.remote(self._request_id)
                for message in pickle.loads(pickled_messages):
                    self._queue.put_nowait(message)
                    if message['type'] in {'http.disconnect', 'websocket.disconnect'}:
                        self._disconnect_message = message
                        return
            except Exception as e:
                self._queue.put_nowait(e)
                return

    async def __call__(self) -> Message:
        """Return the next message once available.

        This will repeatedly return a disconnect message once it's been received.
        """
        if self._queue.empty() and self._disconnect_message is not None:
            return self._disconnect_message
        message = await self._queue.get()
        if isinstance(message, Exception):
            raise message
        return message