import asyncio
import atexit
import concurrent.futures
import errno
import functools
import select
import socket
import sys
import threading
import typing
import warnings
from tornado.gen import convert_yielded
from tornado.ioloop import IOLoop, _Selectable
from typing import (
def _make_current(self) -> None:
    if not self.is_current:
        try:
            self.old_asyncio = asyncio.get_event_loop()
        except (RuntimeError, AssertionError):
            self.old_asyncio = None
        self.is_current = True
    asyncio.set_event_loop(self.asyncio_loop)