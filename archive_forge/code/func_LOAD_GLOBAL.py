import collections
import contextlib
import copy
import dataclasses
import dis
import functools
import importlib
import inspect
import itertools
import linecache
import logging
import operator
import sys
import textwrap
import threading
import traceback
import types
import typing
import weakref
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Type
from unittest.mock import patch
import torch
import torch._logging
from torch._guards import Checkpointable, tracing, TracingContext
from . import (
from .allowed_functions import is_allowed, is_builtin_constant, is_forbidden
from .bytecode_analysis import (
from .bytecode_transformation import (
from .code_context import code_context
from .codegen import PyCodegen
from .current_scope_id import current_scope_id
from .exc import ArgsMismatchError, BackendCompilerFailed, unimplemented, Unsupported
from .funcname_cache import get_funcname
from .guards import GuardBuilder, install_guard
from .output_graph import GraphCompileReason, OutputGraph, OutputGraphState
from .replay_record import DummyModule, ExecutionRecorder
from .resume_execution import ContinueExecutionCache, ReenterWith
from .source import (
from .utils import (
from .variables.base import (
from .variables.builder import VariableBuilder, wrap_fx_proxy
from .variables.builtin import BuiltinVariable
from .variables.constant import ConstantVariable, EnumVariable
from .variables.ctx_manager import (
from .variables.dicts import ConstDictVariable, SetVariable
from .variables.functions import (
from .variables.lists import (
from .variables.misc import (
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
from .variables.torch import TorchVariable
from .variables.user_defined import (
def LOAD_GLOBAL(self, inst):
    if sys.version_info >= (3, 11):
        if inst.arg % 2:
            self.PUSH_NULL(inst)
    name = inst.argval
    if config.replay_record_enabled:
        if name in self.f_globals:
            self.exec_recorder.add_global_var(name, self.f_globals[name])
        else:
            assert name in self.f_builtins
            self.exec_recorder.builtins[name] = self.f_builtins[name]
    if inst.argval == 'AssertionError':
        unimplemented('assert with non-string message')
    if name in self.symbolic_globals:
        variable = self.output.side_effects[self.symbolic_globals[name]]
        self.push(self.output.side_effects.load_global(variable, name))
        return
    try:
        value = self.f_globals[name]
    except KeyError:
        return self.load_builtin(inst)
    source = self.get_global_source(name)
    self.push(VariableBuilder(self, source)(value))