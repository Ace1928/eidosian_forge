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
class InstructionTranslator(InstructionTranslatorBase):
    mutated_closure_cell_contents: Set[str]

    @staticmethod
    def current_tx() -> 'InstructionTranslator':
        return tls.current_tx

    @contextlib.contextmanager
    def set_current_tx(self):
        prior = getattr(tls, 'current_tx', None)
        tls.current_tx = self
        try:
            yield
        finally:
            tls.current_tx = prior

    def __init__(self, instructions: List[Instruction], f_code, f_locals, f_globals, f_builtins, code_options, compiler_fn, one_graph, export, export_constraints, mutated_closure_cell_contents: Set[str], frame_state, speculation_log: SpeculationLog):
        _step_logger()(logging.INFO, f'torchdynamo start tracing {f_code.co_name} {code_options['co_filename']}:{code_options['co_firstlineno']}')
        super().__init__(output=OutputGraph(code_options, compiler_fn, self, export, export_constraints, frame_state, local_scope=f_locals, global_scope=f_globals, f_code=f_code), instructions=instructions, f_locals=f_locals, f_globals=f_globals, f_builtins=f_builtins, code_options=code_options, symbolic_locals={}, symbolic_globals={}, f_code=f_code, export=export, inline_depth=0, speculation_log=speculation_log)
        with tracing(self.output.tracing_context), self.set_current_tx():
            self.one_graph: bool = one_graph
            self.export = export
            self.mutated_closure_cell_contents = mutated_closure_cell_contents
            if self.export:
                assert self.one_graph, 'Export without one graph - something has gone wrong.'
            vars = list(code_options['co_varnames'])
            cells_and_freevars = [x for x in self.cell_and_freevars() if x not in vars]
            vars.extend(cells_and_freevars)
            cells_and_freevars_set = set(cells_and_freevars)
            self.symbolic_locals = {k: variables.LazyVariableTracker.create(f_locals[k], source=LocalSource(k, cell_or_freevar=k in cells_and_freevars_set)) for k in vars if k in f_locals}
            if export:
                self.symbolic_locals = VariableTracker.apply(lambda x: x.realize(), self.symbolic_locals)
            self._freevars_ids = dict()
            for name in self.code_options['co_freevars']:
                if name in f_locals:
                    self._freevars_ids[name] = id(f_locals[name])

    def get_example_value(self, source: Source):
        if isinstance(source, LocalSource):
            return self.f_locals[source.local_name]
        if isinstance(source, GlobalSource):
            return self.f_globals[source.global_name]
        raise KeyError()

    def run(self):
        super().run()

    def match_nested_cell(self, name, cell):
        """Match a cell in this method to one in a function we are inlining"""
        value = cell.cell_contents
        if id(value) != self._freevars_ids.get(name):
            return None
        return self.symbolic_locals[name]

    def should_compile_partial_graph(self):
        return all((b.can_restore() for b in self.block_stack)) and (not self.one_graph) and (self.generic_context_manager_depth == 0)

    def create_call_resume_at(self, inst):
        self.instruction_pointer = None
        if inst.opname == 'RETURN_VALUE':
            return [create_instruction('RETURN_VALUE')]
        reads = livevars_analysis(self.instructions, inst)
        argnames = tuple((k for k in self.symbolic_locals.keys() if k in reads and k not in self.cell_and_freevars()))
        cg = PyCodegen(self)
        null_idxes: List[int] = []
        if sys.version_info >= (3, 11):
            for i, var in enumerate(self.stack):
                if isinstance(var, NullVariable):
                    null_idxes.append(i)
            null_cnt = 0
            for i, var in enumerate(reversed(self.stack)):
                if isinstance(var, NullVariable):
                    for j in range(2, i + 2 - null_cnt):
                        cg.append_output(create_instruction('SWAP', arg=j))
                    cg.extend_output(cg.pop_null())
                    null_cnt += 1
        stack_len = len(self.stack) - len(null_idxes)
        nargs = stack_len + len(argnames)
        name = unique_id(f'__resume_at_{inst.offset}')
        new_code: types.CodeType = ContinueExecutionCache.lookup(self.f_code, self.lineno, inst.offset, tuple((b.target.offset for b in self.block_stack)), stack_len, argnames, tuple((b.resume_fn() for b in self.block_stack)), tuple(null_idxes))
        orig_graphmodule_maybe = code_context.get_context(self.f_code).get('orig_graphmodule', None)
        if orig_graphmodule_maybe is not None:
            code_context.get_context(new_code)['orig_graphmodule'] = orig_graphmodule_maybe
        if new_code.co_freevars:
            cg.make_function_with_closure(name, new_code, True, stack_len)
        else:
            self.output.install_global(name, types.FunctionType(new_code, self.f_globals, name))
            cg.extend_output(cg.load_function_name(name, True, stack_len))
        cg.extend_output([cg.create_load(k) for k in argnames])
        cg.extend_output(create_call_function(nargs, False))
        cg.append_output(create_instruction('RETURN_VALUE'))
        return cg.get_instructions()

    def symbolic_locals_contain_module_class(self):
        for v in self.symbolic_locals.values():
            if isinstance(v, UserDefinedClassVariable) and issubclass(v.as_python_constant(), torch.nn.Module):
                return True
        return False

    def RETURN_VALUE(self, inst):
        if self.output.count_calls() == 0 and (not self.inconsistent_side_effects) and (not self.symbolic_locals_contain_module_class()) and (not self.export):
            raise exc.SkipFrame('because no content in function call')
        self.instruction_pointer = None
        _step_logger()(logging.INFO, f'torchdynamo done tracing {self.f_code.co_name} (RETURN_VALUE)')
        log.debug('RETURN_VALUE triggered compile')
        self.output.compile_subgraph(self, reason=GraphCompileReason('return_value', [self.frame_summary()], graph_break=False), compile_return_value=True)
        self.output.add_output_instructions([create_instruction('RETURN_VALUE')])