import contextlib
import functools
import inspect
import itertools
import logging
import math
import operator
import types
from collections import defaultdict, OrderedDict
from typing import Dict, List
import torch
from torch import sym_float, sym_int
from .. import config, polyfill, variables
from ..exc import (
from ..guards import GuardBuilder, install_guard
from ..replay_record import DummyModule
from ..source import AttrSource, GetItemSource, is_constant_source, TypeSource
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .constant import ConstantVariable
from .ctx_manager import EventVariable, StreamVariable
from .dicts import ConstDictVariable, DefaultDictVariable, SetVariable
from .lists import (
from .tensor import FakeItemVariable, SymNodeVariable, UnspecializedPythonVariable
from .user_defined import UserDefinedVariable
@staticmethod
@functools.lru_cache(None)
def _binops():
    fns = {operator.add: (['__add__', '__radd__', '__iadd__'], operator.iadd), operator.sub: (['__sub__', '__rsub__', '__isub__'], operator.isub), operator.mul: (['__mul__', '__rmul__', '__imul__'], operator.imul), operator.truediv: (['__truediv__', '__rtruediv__', '__itruediv__'], operator.itruediv), operator.floordiv: (['__floordiv__', '__rfloordiv__', '__ifloordiv__'], operator.ifloordiv), operator.mod: (['__mod__', '__rmod__', '__imod__'], operator.imod), pow: (['__pow__', '__rpow__', '__ipow__'], operator.ipow), operator.pow: (['__pow__', '__rpow__', '__ipow__'], operator.ipow), operator.lshift: (['__lshift__', '__rlshift__', '__ilshift__'], operator.ilshift), operator.rshift: (['__rshift__', '__rrshift__', '__irshift__'], operator.irshift)}
    return fns