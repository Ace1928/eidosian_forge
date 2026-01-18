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
def _polyfill_call_impl(name):
    """Create a BuiltinVariable.call_{name} method that inlines through polyfill.{name}"""

    def call_fn(self, tx, *args, **kwargs):
        return tx.inline_user_function_return(variables.UserFunctionVariable(fn), args, kwargs)
    fn = getattr(polyfill, name)
    call_fn.__name__ = f'call_{name}'
    return call_fn