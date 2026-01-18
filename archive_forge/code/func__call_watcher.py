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
def _call_watcher(self_, watcher, event):
    """
        Invoke the given watcher appropriately given an Event object.
        """
    if self_._TRIGGER:
        pass
    elif watcher.onlychanged and (not self_._changed(event)):
        return
    if self_._BATCH_WATCH:
        self_._events.append(event)
        if not any((watcher is w for w in self_._state_watchers)):
            self_._state_watchers.append(watcher)
    else:
        event = self_._update_event_type(watcher, event, self_._TRIGGER)
        with _batch_call_watchers(self_.self_or_cls, enable=watcher.queued, run=False):
            self_._execute_watcher(watcher, (event,))