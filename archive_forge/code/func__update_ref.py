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
def _update_ref(self_, name, ref):
    param_private = self_.self._param__private
    if name in param_private.async_refs:
        param_private.async_refs.pop(name).cancel()
    for _, watcher in param_private.ref_watchers:
        dep_obj = watcher.cls if watcher.inst is None else watcher.inst
        dep_obj.param.unwatch(watcher)
    self_.self._param__private.ref_watchers = []
    refs = dict(self_.self._param__private.refs, **{name: ref})
    deps = {name: resolve_ref(ref) for name, ref in refs.items()}
    self_._setup_refs(deps)
    self_.self._param__private.refs = refs