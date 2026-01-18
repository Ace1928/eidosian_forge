import enum
import dis
import opcode as _opcode
import sys
from marshal import dumps as _dumps
from _pydevd_frame_eval.vendored import bytecode as _bytecode
def _check_arg(self, name, opcode, arg):
    if name == 'EXTENDED_ARG':
        raise ValueError('only concrete instruction can contain EXTENDED_ARG, highlevel instruction can represent arbitrary argument without it')
    if opcode >= _opcode.HAVE_ARGUMENT:
        if arg is UNSET:
            raise ValueError('operation %s requires an argument' % name)
    elif arg is not UNSET:
        raise ValueError('operation %s has no argument' % name)
    if self._has_jump(opcode):
        if not isinstance(arg, (Label, _bytecode.BasicBlock)):
            raise TypeError('operation %s argument type must be Label or BasicBlock, got %s' % (name, type(arg).__name__))
    elif opcode in _opcode.hasfree:
        if not isinstance(arg, (CellVar, FreeVar)):
            raise TypeError('operation %s argument must be CellVar or FreeVar, got %s' % (name, type(arg).__name__))
    elif opcode in _opcode.haslocal or opcode in _opcode.hasname:
        if not isinstance(arg, str):
            raise TypeError('operation %s argument must be a str, got %s' % (name, type(arg).__name__))
    elif opcode in _opcode.hasconst:
        if isinstance(arg, Label):
            raise ValueError('label argument cannot be used in %s operation' % name)
        if isinstance(arg, _bytecode.BasicBlock):
            raise ValueError('block argument cannot be used in %s operation' % name)
    elif opcode in _opcode.hascompare:
        if not isinstance(arg, Compare):
            raise TypeError('operation %s argument type must be Compare, got %s' % (name, type(arg).__name__))
    elif opcode >= _opcode.HAVE_ARGUMENT:
        _check_arg_int(name, arg)