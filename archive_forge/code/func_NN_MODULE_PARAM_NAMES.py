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
def NN_MODULE_PARAM_NAMES(self, guard):
    ref = self.arg_ref(guard)
    value = self.get(guard.name)
    t = type(value)
    keys = {k for k, v in value.named_parameters()}
    code = list()
    code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
    code.append(f'{{k for k, v in {ref}.named_parameters()}} == {keys!r}')
    self._produce_guard_code(guard, code)