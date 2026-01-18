import asyncio
import copy
import datetime as dt
import html
import inspect
import logging
import numbers
import operator
import random
import re
import types
import typing
import warnings
from collections import defaultdict, namedtuple, OrderedDict
from functools import partial, wraps, reduce
from html import escape
from itertools import chain
from operator import itemgetter, attrgetter
from types import FunctionType, MethodType
from contextlib import contextmanager
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from ._utils import (
from inspect import getfullargspec
def _execute_watcher(self, watcher, events):
    if watcher.mode == 'args':
        args, kwargs = (events, {})
    else:
        args, kwargs = ((), {event.name: event.new for event in events})
    if iscoroutinefunction(watcher.fn):
        if async_executor is None:
            raise RuntimeError('Could not execute %s coroutine function. Please register a asynchronous executor on param.parameterized.async_executor, which schedules the function on an event loop.' % watcher.fn)
        async_executor(partial(watcher.fn, *args, **kwargs))
    else:
        try:
            watcher.fn(*args, **kwargs)
        except Skip:
            pass