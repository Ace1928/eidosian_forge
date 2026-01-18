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
def ID_MATCH(self, guard: Guard):
    if isinstance(guard.originating_source, TypeSource):
        return self.TYPE_MATCH(Guard(guard.originating_source.base, GuardBuilder.TYPE_MATCH))
    ref = self.arg_ref(guard)
    val = self.get(guard.name)
    code = f'___check_obj_id({ref}, {self.id_ref(val)})'
    self._produce_guard_code(guard, [code])
    if isinstance(guard.originating_source, LocalSource):
        if isinstance(val, torch.nn.Module):
            local_name = guard.originating_source.local_name
            weak_id = self.lookup_weakrefs(val)
            if weak_id is not None:
                self.id_matched_objs[local_name] = weak_id