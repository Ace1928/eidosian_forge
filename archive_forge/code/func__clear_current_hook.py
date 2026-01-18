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
def _clear_current_hook(self) -> None:
    if self.is_current:
        asyncio.set_event_loop(self.old_asyncio)
        self.is_current = False