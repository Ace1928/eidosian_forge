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
class InliningInstructionTranslator(InstructionTranslatorBase):
    """Trace and inline a called method"""
    symbolic_result: Optional[TensorVariable]

    @classmethod
    def inline_call(cls, parent, func, args, kwargs):
        with patch.dict(counters, {'unimplemented': counters['inline_call']}):
            return cls.inline_call_(parent, func, args, kwargs)

    @staticmethod
    def check_inlineable(func):
        if func.has_self():
            unimplemented('inline with __self__')
        result = skipfiles.check_verbose(func, is_inlined_call=True)
        if result.skipped:
            from torch._dynamo.variables.misc import produce_trampoline_autograd_apply, produce_trampoline_autograd_bwd, produce_trampoline_autograd_fwd
            if hasattr(func.fn, '_origin') and func.fn._origin in [produce_trampoline_autograd_fwd, produce_trampoline_autograd_apply, produce_trampoline_autograd_bwd]:
                return skipfiles.SkipResult(False, 'allowlist in dynamo known function')
            unimplemented(f"'inline in skipfiles: {func.fn.__qualname__} | {func.get_name()} {func.get_filename()}, {result.reason}'")
        if isinstance(func, UserFunctionVariable) and inspect.getattr_static(func.get_function(), '_torchdynamo_disable', False):
            unimplemented(f'call torch._dynamo.disable() wrapped function {func.get_function()}')
        else:
            return result

    @staticmethod
    def inline_call_(parent, func: VariableTracker, args: List[VariableTracker], kwargs):
        assert isinstance(func, (UserFunctionVariable, NestedUserFunctionVariable))
        result = InliningInstructionTranslator.check_inlineable(func)
        assert result.skipped is False
        try:
            sub_locals, closure_cells = func.bind_args(parent, args, kwargs)
        except TypeError as e:
            raise ArgsMismatchError('{reason}.\n  func = {func}, args = {args}, kwargs = {kwargs}'.format(reason=str(e), func=f"'{func.get_name()}' {func.get_filename()}:{func.get_code().co_firstlineno}", args=[arg.python_type() for arg in args], kwargs=kwargs))
        for v in itertools.chain(sub_locals.values(), closure_cells.values()):
            if not isinstance(v, VariableTracker):
                unimplemented(f'unconverted arg {v}')
        code: types.CodeType = func.get_code()
        if code.co_name in ('__setitem__', '__setattr__') and (not (args is not None and len(args) > 0 and isinstance(args[0], variables.CustomizedDictVariable))):
            unimplemented(f'inline {code.co_name}')
        suffix = ''
        if torch._logging._internal.log_state.is_artifact_enabled('output_code'):
            suffix = f'\n{dis.Bytecode(code).dis()}'
        if sys.version_info >= (3, 11):
            cur_inst = parent.current_instruction
            parent_code = parent.f_code
            header = parent.get_line_of_code_header(lineno=cur_inst.positions.lineno)

            def get_trace_call_log_str():
                line = get_instruction_source_311(parent_code, cur_inst).rstrip()
                return f'TRACE inlined call {code.co_name} from {header}\n{line}'
            trace_call_log.debug('%s', LazyString(get_trace_call_log_str))
        log.debug('INLINING %s%s, %s', code, suffix, result.reason)
        if args and isinstance(args[0], NNModuleVariable):
            module = parent.output.get_submodule(args[0].module_key)
            if isinstance(module, torch.fx.GraphModule):
                code_context.get_context(module.forward.__code__)['orig_graphmodule'] = module
        tracer: InliningInstructionTranslator
        if is_generator(code):
            tracer = InliningGeneratorInstructionTranslator(parent, code, sub_locals, parent.symbolic_globals, closure_cells, func)
        else:
            tracer = InliningInstructionTranslator(parent, code, sub_locals, parent.symbolic_globals, closure_cells, func)
        strict_ctx: Any = contextlib.nullcontext()
        if parent.strict_checks_enabled:
            strict_ctx = tracer.strict_translation_mode()
        try:
            with strict_ctx:
                tracer.run()
        except exc.SkipFrame as e:
            msg = f'SKIPPED INLINING {code}: {e}'
            log.debug(msg)
            raise Unsupported(msg) from e
        except Exception as e:
            log.debug('FAILED INLINING %s', code)
            raise
        assert tracer.symbolic_result is not None
        func.export_freevars(parent, tracer)
        if tracer.f_globals is parent.f_globals:
            parent.symbolic_globals.update(tracer.symbolic_globals)
        parent.inconsistent_side_effects |= tracer.inconsistent_side_effects
        log.debug('DONE INLINING %s', code)
        if is_generator(code):
            assert isinstance(tracer, InliningGeneratorInstructionTranslator)
            assert tracer.symbolic_result.as_python_constant() is None
            return ListIteratorVariable(tracer.generated_items, mutable_local=MutableLocal())
        else:
            return tracer.symbolic_result

    def __init__(self, parent: InstructionTranslatorBase, code: types.CodeType, symbolic_locals: Dict[str, VariableTracker], symbolic_globals: Dict[str, VariableTracker], closure_cells: Dict[str, VariableTracker], funcvar: BaseUserFunctionVariable):
        f_globals = funcvar.get_globals()
        f_builtins = f_globals['__builtins__']
        if not isinstance(f_builtins, dict):
            f_builtins = f_builtins.__dict__
        instructions = cleaned_instructions(code)
        propagate_line_nums(instructions)
        super().__init__(output=parent.output, f_locals={}, f_globals=f_globals, f_builtins=f_builtins, symbolic_locals=symbolic_locals, symbolic_globals=symbolic_globals, instructions=instructions, code_options={k: getattr(code, k) for k in dir(code)}, f_code=code, export=parent.export, inline_depth=parent.inline_depth + 1, speculation_log=parent.speculation_log)
        self.parent = parent
        self.symbolic_result = None
        self.closure_cells = closure_cells
        self.nn_module_stack = parent.nn_module_stack.copy()

    @property
    def fake_mode(self):
        return self.parent.fake_mode

    def run_ctx_mgr(self):
        return TracingContext.current_frame(self.parent.frame_summary())

    def STORE_DEREF(self, inst):
        if inst.argval in self.closure_cells:
            cell = self.closure_cells[inst.argval]
            val = self.pop()
            if isinstance(cell, ClosureVariable):
                if not self.output.is_root_tracer():
                    unimplemented('HigherOrderOperator: Mutating a variable not in the current scope (ClosureVariable)')
                self.output.root_tx.symbolic_locals[cell.name] = val
            else:
                self.output.side_effects.store_cell(cell, val)
        else:
            maybe_cell = self.symbolic_locals.get(inst.argval)
            if isinstance(maybe_cell, variables.NewCellVariable):
                self.output.side_effects.store_cell(self.symbolic_locals[inst.argval], self.pop())
            else:
                if maybe_cell is not None and maybe_cell.source.name() not in self.output.root_tx.mutated_closure_cell_contents:
                    self.output.root_tx.mutated_closure_cell_contents.add(maybe_cell.source.name())
                    raise exc.UnspecializeRestartAnalysis()
                unimplemented('write to __closure__ while inlining')

    def LOAD_DEREF(self, inst):
        if inst.argval in self.closure_cells:
            cell = self.closure_cells[inst.argval]
            if isinstance(cell, ClosureVariable):
                self.push(self.output.root_tx.symbolic_locals[cell.name])
            else:
                self.push(self.output.side_effects.load_cell(cell))
        else:
            maybe_sym_local = self.symbolic_locals.get(inst.argval, None)
            if isinstance(maybe_sym_local, variables.NewCellVariable):
                self.push(self.output.side_effects.load_cell(maybe_sym_local))
            else:
                super().LOAD_DEREF(inst)

    def LOAD_CLOSURE(self, inst):
        assert inst.argval in self.cell_and_freevars()
        if inst.argval in self.closure_cells:
            self.push(self.closure_cells[inst.argval])
        else:
            self.push(InlinedClosureVariable(name=inst.argval))

    def check_replace_is_safe(self, oldvar):
        if not is_side_effect_safe(oldvar.mutable_local):
            unimplemented('HigherOrderOperator: Mutating a variable not in the current scope (replace_all)')

    def replace_all(self, oldvar: VariableTracker, newvar: VariableTracker):
        self.check_replace_is_safe(oldvar)
        newvar = super().replace_all(oldvar, newvar)
        translator: InstructionTranslatorBase = self
        while hasattr(translator, 'parent'):
            translator = translator.parent
            translator.update_locals_and_stack(oldvar, newvar)
        return newvar

    def should_compile_partial_graph(self):
        return False

    def create_call_resume_at(self, offset):
        unimplemented('cant resume while inlining')

    def RETURN_VALUE(self, inst):
        self.symbolic_result = self.pop()
        self.instruction_pointer = None