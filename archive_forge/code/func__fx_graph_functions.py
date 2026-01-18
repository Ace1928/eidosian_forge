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
def _fx_graph_functions():
    fns = {operator.pos, operator.neg, operator.not_, operator.invert, operator.pow, operator.mul, operator.matmul, operator.floordiv, operator.truediv, operator.mod, operator.add, operator.lt, operator.gt, operator.ge, operator.le, operator.ne, operator.eq, operator.sub, operator.getitem, operator.lshift, operator.rshift, operator.and_, operator.or_, operator.xor, operator.ipow, operator.imul, operator.imatmul, operator.ifloordiv, operator.itruediv, operator.imod, operator.iadd, operator.isub, operator.ilshift, operator.irshift, operator.iand, operator.ixor, operator.ior}
    return fns