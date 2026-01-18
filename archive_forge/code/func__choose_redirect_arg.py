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
def _choose_redirect_arg(self, follow_redirects: bool | None, allow_redirects: bool | None) -> bool | httpx._client.UseClientDefault:
    redirect: bool | httpx._client.UseClientDefault = httpx._client.USE_CLIENT_DEFAULT
    if allow_redirects is not None:
        message = 'The `allow_redirects` argument is deprecated. Use `follow_redirects` instead.'
        warnings.warn(message, DeprecationWarning)
        redirect = allow_redirects
    if follow_redirects is not None:
        redirect = follow_redirects
    elif allow_redirects is not None and follow_redirects is not None:
        raise RuntimeError('Cannot use both `allow_redirects` and `follow_redirects`.')
    return redirect