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
def TENSOR_MATCH(self, guard: Guard, value=None):
    if guard.is_nn_module():
        self.ID_MATCH(guard)
    else:
        if isinstance(value, TensorWeakRef):
            value = value()
        value = value if value is not None else self.get(guard.name)
        assert isinstance(value, torch.Tensor)
        tensor_name = self.arg_ref(guard)
        code: List[str] = list()
        if self.check_fn_manager.output_graph.export:
            self.TYPE_MATCH(guard)
            terms = ['dtype', 'device', 'requires_grad', 'ndimension()']
            for term in terms:
                real_value = self.get(tensor_name + '.' + term)
                if istype(real_value, (torch.device, torch.dtype)):
                    code.append(f'str({tensor_name}.{term}) == {str(real_value)!r}')
                else:
                    code.append(f'{tensor_name}.{term} == {real_value}')
        else:
            self.tensor_check_names.append(tensor_name)
            self.tensor_check_examples.append(value)
            self.tensor_check_guards.append(guard)
        assert guard.source is not None
        static, reason = tensor_always_has_static_shape(value, is_tensor=True, guard_source=guard.source)
        if not static:
            if hasattr(value, '_dynamo_dynamic_indices'):
                code.append(f"(({tensor_name}._dynamo_dynamic_indices.issubset({value._dynamo_dynamic_indices})) if hasattr({tensor_name}, '_dynamo_dynamic_indices') else True)")
            else:
                code.append(f"hasattr({tensor_name}, '_dynamo_dynamic_indices') == False")
        if len(code) > 0:
            self._produce_guard_code(guard, code)