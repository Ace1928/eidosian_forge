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
class ASGIAppReplicaWrapper:
    """Provides a common wrapper for replicas running an ASGI app."""

    def __init__(self, app: ASGIApp):
        self._asgi_app = app
        self._serve_asgi_lifespan = LifespanOn(Config(self._asgi_app, lifespan='on'))
        self._serve_asgi_lifespan.logger = logger

    async def _run_asgi_lifespan_startup(self):
        from ray.serve._private.logging_utils import LoggingContext
        with LoggingContext(self._serve_asgi_lifespan.logger, level=logging.WARNING):
            await self._serve_asgi_lifespan.startup()
            if self._serve_asgi_lifespan.should_exit:
                raise RuntimeError('ASGI lifespan startup failed. Check replica logs for details.')

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> Optional[ASGIApp]:
        """Calls into the wrapped ASGI app."""
        await self._asgi_app(scope, receive, send)

    async def __del__(self):
        from ray.serve._private.logging_utils import LoggingContext
        with LoggingContext(self._serve_asgi_lifespan.logger, level=logging.WARNING):
            await self._serve_asgi_lifespan.shutdown()