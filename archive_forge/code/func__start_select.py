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
def _start_select(self) -> None:
    with self._select_cond:
        assert self._select_args is None
        self._select_args = (list(self._readers.keys()), list(self._writers.keys()))
        self._select_cond.notify()