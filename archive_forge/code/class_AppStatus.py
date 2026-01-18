import io
import logging
import re
from datetime import datetime, timezone
from functools import partial
from typing import (
import anyio
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.responses import Response
from starlette.types import Receive, Scope, Send
class AppStatus:
    """helper for monkey-patching the signal-handler of uvicorn"""
    should_exit = False
    should_exit_event: Union[anyio.Event, None] = None

    @staticmethod
    def handle_exit(*args, **kwargs):
        AppStatus.should_exit = True
        if AppStatus.should_exit_event is not None:
            AppStatus.should_exit_event.set()
        original_handler(*args, **kwargs)