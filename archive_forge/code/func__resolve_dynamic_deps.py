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
def _resolve_dynamic_deps(self, obj, dynamic_dep, param_dep, attribute):
    """
        If a subobject whose parameters are being depended on changes
        we should only trigger events if the actual parameter values
        of the new object differ from those on the old subobject,
        therefore we accumulate parameters to compare on a subobject
        change event.

        Additionally we need to make sure to notify the parent object
        if a subobject changes so the dependencies can be
        reinitialized so we return a callback which updates the
        dependencies.
        """
    subobj = obj
    subobjs = [obj]
    for subpath in dynamic_dep.spec.split('.')[:-1]:
        subobj = getattr(subobj, subpath.split(':')[0], None)
        subobjs.append(subobj)
    dep_obj = param_dep.cls if param_dep.inst is None else param_dep.inst
    if dep_obj not in subobjs[:-1]:
        return (None, None, param_dep.what)
    depth = subobjs.index(dep_obj)
    callback = None
    if depth > 0:

        def callback(*events):
            """
                If a subobject changes, we need to notify the main
                object to update the dependencies.
                """
            obj.param._update_deps(attribute)
    p = '.'.join(dynamic_dep.spec.split(':')[0].split('.')[depth + 1:])
    if p == 'param':
        subparams = [sp for sp in list(subobjs[-1].param)]
    else:
        subparams = [p]
    if ':' in dynamic_dep.spec:
        what = dynamic_dep.spec.split(':')[-1]
    else:
        what = param_dep.what
    return (subparams, callback, what)