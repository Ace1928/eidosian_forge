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
def add_code_part(code, guard, log_only=False):
    extra = ''
    if guard.user_stack:
        for fs in reversed(guard.user_stack):
            if fs.filename not in uninteresting_files():
                extra = f'  # {format_frame(fs, line=True)}'
                break
    elif guard.stack:
        extra = f'  # {format_frame(guard.stack.summary()[-1])}'
    guards_log.debug('%s', f'{code:<60}{extra}')
    if verbose_guards_log.isEnabledFor(logging.DEBUG):
        maybe_stack = ''
        maybe_user_stack = ''
        if guard is not None:
            if guard.stack:
                maybe_stack = f'\nStack:\n{''.join(guard.stack.format())}'
            if guard.user_stack:
                maybe_user_stack = f'\nUser stack:\n{''.join(guard.user_stack.format())}'
        verbose_guards_log.debug('Guard: %s%s%s', code, maybe_stack, maybe_user_stack)
    if not log_only:
        code_parts.append(code)
        verbose_code_parts.append(f'{code:<60}{extra}')