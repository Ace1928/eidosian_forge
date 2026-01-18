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
class GunicornUVLoopWebWorker(GunicornWebWorker):

    def init_process(self) -> None:
        import uvloop
        asyncio.get_event_loop().close()
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        super().init_process()