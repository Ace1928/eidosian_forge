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
def _update_deps(self_, attribute=None, init=False):
    obj = self_.self
    init_methods = []
    for method, queued, on_init, constant, dynamic in type(obj).param._depends['watch']:
        dynamic = [d for d in dynamic if attribute is None or d.spec.split('.')[0] == attribute]
        if init:
            constant_grouped = defaultdict(list)
            for dep in _resolve_mcs_deps(obj, constant, []):
                constant_grouped[id(dep.inst), id(dep.cls), dep.what].append((None, dep))
            for group in constant_grouped.values():
                self_._watch_group(obj, method, queued, group)
            m = getattr(self_.self, method)
            if on_init and m not in init_methods:
                init_methods.append(m)
        elif dynamic:
            for w in obj._param__private.dynamic_watchers.pop(method, []):
                (w.cls if w.inst is None else w.inst).param.unwatch(w)
        else:
            continue
        grouped = defaultdict(list)
        for ddep in dynamic:
            for dep in _resolve_mcs_deps(obj, [], [ddep]):
                grouped[id(dep.inst), id(dep.cls), dep.what].append((ddep, dep))
        for group in grouped.values():
            watcher = self_._watch_group(obj, method, queued, group, attribute)
            obj._param__private.dynamic_watchers[method].append(watcher)
    for m in init_methods:
        m()