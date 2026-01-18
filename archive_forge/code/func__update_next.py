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
def _update_next(self, current_time: float) -> None:
    callback_time_sec = self.callback_time / 1000.0
    if self.jitter:
        callback_time_sec *= 1 + self.jitter * (random.random() - 0.5)
    if self._next_timeout <= current_time:
        self._next_timeout += (math.floor((current_time - self._next_timeout) / callback_time_sec) + 1) * callback_time_sec
    else:
        self._next_timeout += callback_time_sec