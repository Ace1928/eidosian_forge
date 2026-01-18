from __future__ import annotations
import ast
import builtins
import collections
import dataclasses
import enum
import functools
import importlib
import inspect
import itertools
import logging
import math
import os
import re
import sys
import textwrap
import types
import weakref
from inspect import currentframe, getframeinfo
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from weakref import ReferenceType
import torch
import torch.utils._device
from torch._dynamo.source import (
from torch._guards import (
from torch.fx.experimental.symbolic_shapes import (
from torch.utils._traceback import format_frame, report_compile_source_on_error
from torch.utils.weak import TensorWeakRef
from . import config, convert_frame, exc, mutation_guard
from .eval_frame import set_guard_error_hook
from .source import DefaultsSource, LocalSource, TypeSource
from .types import GuardedCode, GuardFail, GuardFn  # noqa: F401
from .utils import (
def EQUALS_MATCH(self, guard: Guard):
    ref = self.arg_ref(guard)
    val = self.get(guard.name)
    t = type(val)
    if np:
        np_types: Tuple[Type[Any], ...] = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64)
    else:
        np_types = ()
    ok_types = (int, float, bool, type(None), str, type, list, tuple, set, slice, frozenset, range, torch.Size, torch.device, torch.dtype, *np_types)
    if istype(val, dict):
        assert all((istype(x, ok_types) for x in itertools.chain(val.keys(), val.values())))
    else:
        assert istype(val, ok_types), t.__name__
    if istype(val, float) and math.isnan(val):
        code = list()
        code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
        code.append(f'__math_isnan({ref})')
        self._produce_guard_code(guard, code)
        return
    code = list()
    if istype(val, (list, tuple)):
        self.LIST_LENGTH(guard)
        for idx, elem in enumerate(val):
            code.append(f'___check_type_id({ref}[{idx}], {self.id_ref(type(elem))})')
    else:
        code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
    if istype(val, torch.Size):
        val = tuple(val)
    code.append(f'{ref} == {val!r}')
    self._produce_guard_code(guard, code)