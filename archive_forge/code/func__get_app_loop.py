from __future__ import annotations
import asyncio
import queue
import sys
import threading
import time
from contextlib import contextmanager
from typing import Generator, TextIO, cast
from .application import get_app_session, run_in_terminal
from .output import Output
def _get_app_loop(self) -> asyncio.AbstractEventLoop | None:
    """
        Return the event loop for the application currently running in our
        `AppSession`.
        """
    app = self.app_session.app
    if app is None:
        return None
    return app.loop