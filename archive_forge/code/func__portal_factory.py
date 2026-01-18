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
@contextlib.contextmanager
def _portal_factory(self) -> typing.Generator[anyio.abc.BlockingPortal, None, None]:
    if self.portal is not None:
        yield self.portal
    else:
        with anyio.from_thread.start_blocking_portal(**self.async_backend) as portal:
            yield portal