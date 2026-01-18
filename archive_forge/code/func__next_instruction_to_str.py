import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
def _next_instruction_to_str(self, line_to_contents):
    if self.instructions:
        ret = self._lookahead()
        if ret:
            return ret
    msg = self._create_msg_part
    instruction = self.instructions.pop(0)
    if instruction.opname in 'RESUME':
        return None
    if instruction.opname in ('LOAD_GLOBAL', 'LOAD_FAST', 'LOAD_CONST', 'LOAD_NAME'):
        next_instruction = self.instructions[0]
        if next_instruction.opname in ('STORE_FAST', 'STORE_NAME'):
            self.instructions.pop(0)
            return (msg(next_instruction), msg(next_instruction, ' = '), msg(instruction))
        if next_instruction.opname == 'RETURN_VALUE':
            self.instructions.pop(0)
            return (msg(instruction, 'return ', line=self.min_line(instruction)), msg(instruction))
        if next_instruction.opname == 'RAISE_VARARGS' and next_instruction.argval == 1:
            self.instructions.pop(0)
            return (msg(instruction, 'raise ', line=self.min_line(instruction)), msg(instruction))
    if instruction.opname == 'LOAD_CONST':
        if inspect.iscode(instruction.argval):
            code_line_to_contents = _Disassembler(instruction.argval, self.firstlineno, self.level + 1).build_line_to_contents()
            for contents in code_line_to_contents.values():
                contents.insert(0, '    ')
            for line, contents in code_line_to_contents.items():
                line_to_contents.setdefault(line, []).extend(contents)
            return msg(instruction, 'LOAD_CONST(code)')
    if instruction.opname == 'RAISE_VARARGS':
        if instruction.argval == 0:
            return msg(instruction, 'raise')
    if instruction.opname == 'SETUP_FINALLY':
        return msg(instruction, ('try(', instruction.argrepr, '):'))
    if instruction.argrepr:
        return msg(instruction, (instruction.opname, '(', instruction.argrepr, ')'))
    if instruction.argval:
        return msg(instruction, '%s{%s}' % (instruction.opname, instruction.argval))
    return msg(instruction, instruction.opname)