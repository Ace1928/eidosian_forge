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
def _call_min_max(self, tx, *args):
    if len(args) == 1 and args[0].has_unpack_var_sequence(tx):
        items = args[0].unpack_var_sequence(tx)
        return self._call_min_max_seq(tx, items)
    elif len(args) == 2:
        return self._call_min_max_binary(tx, args[0], args[1])
    elif len(args) > 2:
        return self._call_min_max_seq(tx, args)