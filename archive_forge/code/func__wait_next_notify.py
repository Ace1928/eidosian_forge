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
def _wait_next_notify(self) -> 'asyncio.Future[bool]':
    self._notify_waiter_done()
    loop = self.loop
    assert loop is not None
    self._notify_waiter = waiter = loop.create_future()
    self.loop.call_later(1.0, self._notify_waiter_done, waiter)
    return waiter