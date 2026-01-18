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
def _parse_dependency_spec(spec):
    """
    Parses param.depends specifications into three components:

    1. The dotted path to the sub-object
    2. The attribute being depended on, i.e. either a parameter or method
    3. The parameter attribute being depended on
    """
    assert spec.count(':') <= 1
    spec = spec.strip()
    m = re.match('(?P<path>[^:]*):?(?P<what>.*)', spec)
    what = m.group('what')
    path = '.' + m.group('path')
    m = re.match('(?P<obj>.*)(\\.)(?P<attr>.*)', path)
    obj = m.group('obj')
    attr = m.group('attr')
    return (obj or None, attr, what or 'value')