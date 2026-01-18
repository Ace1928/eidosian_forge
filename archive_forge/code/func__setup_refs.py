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
def _setup_refs(self_, refs):
    groups = defaultdict(list)
    for pname, subrefs in refs.items():
        for p in subrefs:
            if isinstance(p, Parameter):
                groups[p.owner].append((pname, p.name))
            else:
                for sp in extract_dependencies(p):
                    groups[sp.owner].append((pname, sp.name))
    for owner, pnames in groups.items():
        refnames, pnames = zip(*pnames)
        self_.self._param__private.ref_watchers.append((refnames, owner.param._watch(self_._sync_refs, list(set(pnames)), precedence=-1)))