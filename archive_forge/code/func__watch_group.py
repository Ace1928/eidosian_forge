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
def _watch_group(self_, obj, name, queued, group, attribute=None):
    """
        Sets up a watcher for a group of dependencies. Ensures that
        if the dependency was dynamically generated we check whether
        a subobject change event actually causes a value change and
        that we update the existing watchers, i.e. clean up watchers
        on the old subobject and create watchers on the new subobject.
        """
    dynamic_dep, param_dep = group[0]
    dep_obj = param_dep.cls if param_dep.inst is None else param_dep.inst
    params = []
    for _, g in group:
        if g.name not in params:
            params.append(g.name)
    if dynamic_dep is None:
        subparams, callback, what = (None, None, param_dep.what)
    else:
        subparams, callback, what = self_._resolve_dynamic_deps(obj, dynamic_dep, param_dep, attribute)
    mcaller = _m_caller(obj, name, what, subparams, callback)
    return dep_obj.param._watch(mcaller, params, param_dep.what, queued=queued, precedence=-1)