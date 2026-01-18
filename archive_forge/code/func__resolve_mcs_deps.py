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
def _resolve_mcs_deps(obj, resolved, dynamic, intermediate=True):
    """
    Resolves constant and dynamic parameter dependencies previously
    obtained using the _params_depended_on function. Existing resolved
    dependencies are updated with a supplied parameter instance while
    dynamic dependencies are resolved if possible.
    """
    dependencies = []
    for dep in resolved:
        if not issubclass(type(obj), dep.cls):
            dependencies.append(dep)
            continue
        inst = obj if dep.inst is None else dep.inst
        dep = PInfo(inst=inst, cls=dep.cls, name=dep.name, pobj=inst.param[dep.name], what=dep.what)
        dependencies.append(dep)
    for dep in dynamic:
        subresolved, _ = obj.param._spec_to_obj(dep.spec, intermediate=intermediate)
        for subdep in subresolved:
            if isinstance(subdep, PInfo):
                dependencies.append(subdep)
            else:
                dependencies += _params_depended_on(subdep, intermediate=intermediate)[0]
    return dependencies