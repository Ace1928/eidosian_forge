import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
def _get_except_target_info(instructions, exception_end_instruction_index, offset_to_instruction_idx):
    next_3 = [j_instruction.opname for j_instruction in instructions[exception_end_instruction_index:exception_end_instruction_index + 3]]
    if next_3 == ['POP_TOP', 'POP_TOP', 'POP_TOP']:
        for pop_except_instruction in instructions[exception_end_instruction_index + 3:]:
            if pop_except_instruction.opname == 'POP_EXCEPT':
                except_end_instruction = pop_except_instruction
                return _TargetInfo(except_end_instruction)
    elif next_3 and next_3[0] == 'DUP_TOP':
        iter_in = instructions[exception_end_instruction_index + 1:]
        for jump_if_not_exc_instruction in iter_in:
            if jump_if_not_exc_instruction.opname == 'JUMP_IF_NOT_EXC_MATCH':
                except_end_instruction = instructions[offset_to_instruction_idx[jump_if_not_exc_instruction.argval]]
                return _TargetInfo(except_end_instruction, jump_if_not_exc_instruction)
        else:
            return None
    else:
        return None