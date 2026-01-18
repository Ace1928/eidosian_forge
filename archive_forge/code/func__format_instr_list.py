import sys
import textwrap
import types
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
def _format_instr_list(block, labels, lineno):
    instr_list = []
    for instr in block:
        if not isinstance(instr, Label):
            if isinstance(instr, ConcreteInstr):
                cls_name = 'ConcreteInstr'
            else:
                cls_name = 'Instr'
            arg = instr.arg
            if arg is not UNSET:
                if isinstance(arg, Label):
                    arg = labels[arg]
                elif isinstance(arg, BasicBlock):
                    arg = labels[id(arg)]
                else:
                    arg = repr(arg)
                if lineno:
                    text = '%s(%r, %s, lineno=%s)' % (cls_name, instr.name, arg, instr.lineno)
                else:
                    text = '%s(%r, %s)' % (cls_name, instr.name, arg)
            elif lineno:
                text = '%s(%r, lineno=%s)' % (cls_name, instr.name, instr.lineno)
            else:
                text = '%s(%r)' % (cls_name, instr.name)
        else:
            text = labels[instr]
        instr_list.append(text)
    return '[%s]' % ',\n '.join(instr_list)