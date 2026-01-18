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
def _batch_call_watchers(self_):
    """
        Batch call a set of watchers based on the parameter value
        settings in kwargs using the queued Event and watcher objects.
        """
    while self_._events:
        event_dict = OrderedDict([((event.name, event.what), event) for event in self_._events])
        watchers = self_._state_watchers[:]
        self_._events = []
        self_._state_watchers = []
        for watcher in sorted(watchers, key=lambda w: w.precedence):
            events = [self_._update_event_type(watcher, event_dict[name, watcher.what], self_._TRIGGER) for name in watcher.parameter_names if (name, watcher.what) in event_dict]
            with _batch_call_watchers(self_.self_or_cls, enable=watcher.queued, run=False):
                self_._execute_watcher(watcher, events)