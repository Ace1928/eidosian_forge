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
@dataclasses.dataclass
class SpeculationLog:
    """
    SpeculationLog replaces the prior copy_graphstate/restore_graphstate
    checkpointing.  Rather than saving/restoring state, we restart the
    dynamo conversion process over from the beginning -- but when we
    hit the start of the speculation that failed, we instead generate
    a graph break.
    """
    entries: List[SpeculationEntry] = dataclasses.field(default_factory=list)
    index: int = 0

    def restart(self):
        self.index = 0

    def clear(self):
        self.entries.clear()
        self.index = 0

    def next(self, filename: str, lineno: int, instruction_pointer) -> SpeculationEntry:
        """
        Lookup or create a SpeculationEntry() that is shared across
        RestartAnalysis calls.  Args are used only for debug checks.
        """
        if len(self.entries) == self.index:
            self.entries.append(SpeculationEntry(filename, lineno, instruction_pointer))
        entry = self.entries[self.index]
        self.index += 1
        assert entry.instruction_pointer == instruction_pointer and entry.filename == filename and (entry.lineno == lineno), textwrap.dedent(f'\n            SpecuationLog diverged at {self.index} of {len(self.entries)}:\n            - Run1: {entry.filename}:{entry.lineno} (ip={entry.instruction_pointer})\n            - Run2: {filename}:{lineno} (ip={instruction_pointer})\n            Please submit a bug report.\n            ')
        return entry