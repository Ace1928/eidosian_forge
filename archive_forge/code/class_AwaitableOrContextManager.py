from __future__ import annotations
import asyncio
import functools
import re
import sys
import typing
from contextlib import contextmanager
from starlette.types import Scope
class AwaitableOrContextManager(typing.Awaitable[T_co], typing.AsyncContextManager[T_co], typing.Protocol[T_co]):
    ...