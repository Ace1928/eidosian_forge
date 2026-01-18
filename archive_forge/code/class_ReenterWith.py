import copy
import dataclasses
import sys
import types
from typing import Any, cast, Dict, List, Optional, Tuple
from .bytecode_transformation import (
from .utils import ExactWeakKeyDictionary
@dataclasses.dataclass(frozen=True)
class ReenterWith:
    stack_index: int
    target_values: Optional[Tuple[Any, ...]] = None

    def try_except(self, code_options, cleanup: List[Instruction]):
        """
        Codegen based off of:
        load args
        enter context
        try:
            (rest)
        finally:
            exit context
        """
        load_args = []
        if self.target_values:
            load_args = [create_instruction('LOAD_CONST', argval=val) for val in self.target_values]
        ctx_name = unique_id(f'___context_manager_{self.stack_index}')
        if ctx_name not in code_options['co_varnames']:
            code_options['co_varnames'] += (ctx_name,)
        for name in ['__enter__', '__exit__']:
            if name not in code_options['co_names']:
                code_options['co_names'] += (name,)
        except_jump_target = create_instruction('NOP' if sys.version_info < (3, 11) else 'PUSH_EXC_INFO')
        cleanup_complete_jump_target = create_instruction('NOP')
        setup_finally = [*load_args, *create_call_function(len(load_args), True), create_instruction('STORE_FAST', argval=ctx_name), create_instruction('LOAD_FAST', argval=ctx_name), create_instruction('LOAD_METHOD', argval='__enter__'), *create_call_method(0), create_instruction('POP_TOP')]
        if sys.version_info < (3, 11):
            setup_finally.append(create_instruction('SETUP_FINALLY', target=except_jump_target))
        else:
            exn_tab_begin = create_instruction('NOP')
            exn_tab_end = create_instruction('NOP')
            exn_tab_begin.exn_tab_entry = InstructionExnTabEntry(exn_tab_begin, exn_tab_end, except_jump_target, self.stack_index + 1, False)
            setup_finally.append(exn_tab_begin)

        def create_reset():
            return [create_instruction('LOAD_FAST', argval=ctx_name), create_instruction('LOAD_METHOD', argval='__exit__'), create_instruction('LOAD_CONST', argval=None), create_dup_top(), create_dup_top(), *create_call_method(3), create_instruction('POP_TOP')]
        if sys.version_info < (3, 9):
            epilogue = [create_instruction('POP_BLOCK'), create_instruction('BEGIN_FINALLY'), except_jump_target, *create_reset(), create_instruction('END_FINALLY')]
        elif sys.version_info < (3, 11):
            epilogue = [create_instruction('POP_BLOCK'), *create_reset(), create_instruction('JUMP_FORWARD', target=cleanup_complete_jump_target), except_jump_target, *create_reset(), create_instruction('RERAISE'), cleanup_complete_jump_target]
        else:
            finally_exn_tab_end = create_instruction('RERAISE', arg=0)
            finally_exn_tab_target = create_instruction('COPY', arg=3)
            except_jump_target.exn_tab_entry = InstructionExnTabEntry(except_jump_target, finally_exn_tab_end, finally_exn_tab_target, self.stack_index + 2, True)
            epilogue = [exn_tab_end, *create_reset(), create_instruction('JUMP_FORWARD', target=cleanup_complete_jump_target), except_jump_target, *create_reset(), finally_exn_tab_end, finally_exn_tab_target, create_instruction('POP_EXCEPT'), create_instruction('RERAISE', arg=1), cleanup_complete_jump_target]
        cleanup[:] = epilogue + cleanup
        return setup_finally

    def __call__(self, code_options, cleanup):
        """
        Codegen based off of:
        with ctx(args):
            (rest)
        """
        load_args = []
        if self.target_values:
            load_args = [create_instruction('LOAD_CONST', argval=val) for val in self.target_values]
        if sys.version_info < (3, 9):
            with_cleanup_start = create_instruction('WITH_CLEANUP_START')
            begin_finally = create_instruction('BEGIN_FINALLY')
            cleanup[:] = [create_instruction('POP_BLOCK'), begin_finally, with_cleanup_start, create_instruction('WITH_CLEANUP_FINISH'), create_instruction('END_FINALLY')] + cleanup
            return ([*load_args, create_instruction('CALL_FUNCTION', arg=len(load_args)), create_instruction('SETUP_WITH', target=with_cleanup_start), create_instruction('POP_TOP')], None)
        elif sys.version_info < (3, 11):
            with_except_start = create_instruction('WITH_EXCEPT_START')
            pop_top_after_with_except_start = create_instruction('POP_TOP')
            cleanup_complete_jump_target = create_instruction('NOP')
            cleanup[:] = [create_instruction('POP_BLOCK'), create_instruction('LOAD_CONST', argval=None), create_instruction('DUP_TOP'), create_instruction('DUP_TOP'), create_instruction('CALL_FUNCTION', arg=3), create_instruction('POP_TOP'), create_instruction('JUMP_FORWARD', target=cleanup_complete_jump_target), with_except_start, create_instruction('POP_JUMP_IF_TRUE', target=pop_top_after_with_except_start), create_instruction('RERAISE'), pop_top_after_with_except_start, create_instruction('POP_TOP'), create_instruction('POP_TOP'), create_instruction('POP_EXCEPT'), create_instruction('POP_TOP'), cleanup_complete_jump_target] + cleanup
            return ([*load_args, create_instruction('CALL_FUNCTION', arg=len(load_args)), create_instruction('SETUP_WITH', target=with_except_start), create_instruction('POP_TOP')], None)
        else:
            pop_top_after_with_except_start = create_instruction('POP_TOP')
            cleanup_complete_jump_target = create_instruction('NOP')

            def create_load_none():
                return create_instruction('LOAD_CONST', argval=None)
            exn_tab_1_begin = create_instruction('POP_TOP')
            exn_tab_1_end = create_instruction('NOP')
            exn_tab_1_target = create_instruction('PUSH_EXC_INFO')
            exn_tab_2_end = create_instruction('RERAISE', arg=2)
            exn_tab_2_target = create_instruction('COPY', arg=3)
            exn_tab_1_begin.exn_tab_entry = InstructionExnTabEntry(exn_tab_1_begin, exn_tab_1_end, exn_tab_1_target, self.stack_index + 1, True)
            exn_tab_1_target.exn_tab_entry = InstructionExnTabEntry(exn_tab_1_target, exn_tab_2_end, exn_tab_2_target, self.stack_index + 3, True)
            pop_top_after_with_except_start.exn_tab_entry = InstructionExnTabEntry(pop_top_after_with_except_start, pop_top_after_with_except_start, exn_tab_2_target, self.stack_index + 3, True)
            cleanup[:] = [exn_tab_1_end, create_load_none(), create_load_none(), create_load_none(), *create_call_function(2, False), create_instruction('POP_TOP'), create_instruction('JUMP_FORWARD', target=cleanup_complete_jump_target), exn_tab_1_target, create_instruction('WITH_EXCEPT_START'), create_instruction('POP_JUMP_FORWARD_IF_TRUE', target=pop_top_after_with_except_start), exn_tab_2_end, exn_tab_2_target, create_instruction('POP_EXCEPT'), create_instruction('RERAISE', arg=1), pop_top_after_with_except_start, create_instruction('POP_EXCEPT'), create_instruction('POP_TOP'), create_instruction('POP_TOP'), cleanup_complete_jump_target] + cleanup
            return ([*load_args, *create_call_function(len(load_args), True), create_instruction('BEFORE_WITH'), exn_tab_1_begin], exn_tab_1_target)