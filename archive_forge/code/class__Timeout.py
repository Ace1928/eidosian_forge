import asyncio
import concurrent.futures
import datetime
import functools
import numbers
import os
import sys
import time
import math
import random
import warnings
from inspect import isawaitable
from tornado.concurrent import (
from tornado.log import app_log
from tornado.util import Configurable, TimeoutError, import_object
import typing
from typing import Union, Any, Type, Optional, Callable, TypeVar, Tuple, Awaitable
class _Timeout(object):
    """An IOLoop timeout, a UNIX timestamp and a callback"""
    __slots__ = ['deadline', 'callback', 'tdeadline']

    def __init__(self, deadline: float, callback: Callable[[], None], io_loop: IOLoop) -> None:
        if not isinstance(deadline, numbers.Real):
            raise TypeError('Unsupported deadline %r' % deadline)
        self.deadline = deadline
        self.callback = callback
        self.tdeadline = (deadline, next(io_loop._timeout_counter))

    def __lt__(self, other: '_Timeout') -> bool:
        return self.tdeadline < other.tdeadline

    def __le__(self, other: '_Timeout') -> bool:
        return self.tdeadline <= other.tdeadline