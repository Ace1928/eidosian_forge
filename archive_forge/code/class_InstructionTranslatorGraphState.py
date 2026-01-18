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
class InstructionTranslatorGraphState(NamedTuple):
    output: OutputGraphState
    symbolic_locals: Dict[str, VariableTracker]
    stack: List[VariableTracker]
    block_stack: List[BlockStackEntry]
    instruction_pointer: Optional[int]
    current_instruction: Instruction
    next_instruction: Optional[Instruction]
    lineno: int

    def diff(self, other: 'InstructionTranslatorGraphState') -> Optional[str]:
        for k in self._fields:
            if k == 'output':
                return self.output.diff(other.output, prefix=f'{k}.')
            sv = getattr(self, k)
            ov = getattr(other, k)
            if sv != ov:
                return f'{k} mismatch: {sv} != {ov}'
        return None