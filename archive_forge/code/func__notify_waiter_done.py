import asyncio
import os
import re
import signal
import sys
from types import FrameType
from typing import Any, Awaitable, Callable, Optional, Union  # noqa
from gunicorn.config import AccessLogFormat as GunicornAccessLogFormat
from gunicorn.workers import base
from aiohttp import web
from .helpers import set_result
from .web_app import Application
from .web_log import AccessLogger
def _notify_waiter_done(self, waiter: Optional['asyncio.Future[bool]']=None) -> None:
    if waiter is None:
        waiter = self._notify_waiter
    if waiter is not None:
        set_result(waiter, True)
    if waiter is self._notify_waiter:
        self._notify_waiter = None