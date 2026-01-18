import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _format_code_info(co):
    lines = []
    lines.append('Name:              %s' % co.co_name)
    lines.append('Filename:          %s' % co.co_filename)
    lines.append('Argument count:    %s' % co.co_argcount)
    lines.append('Positional-only arguments: %s' % co.co_posonlyargcount)
    lines.append('Kw-only arguments: %s' % co.co_kwonlyargcount)
    lines.append('Number of locals:  %s' % co.co_nlocals)
    lines.append('Stack size:        %s' % co.co_stacksize)
    lines.append('Flags:             %s' % pretty_flags(co.co_flags))
    if co.co_consts:
        lines.append('Constants:')
        for i_c in enumerate(co.co_consts):
            lines.append('%4d: %r' % i_c)
    if co.co_names:
        lines.append('Names:')
        for i_n in enumerate(co.co_names):
            lines.append('%4d: %s' % i_n)
    if co.co_varnames:
        lines.append('Variable names:')
        for i_n in enumerate(co.co_varnames):
            lines.append('%4d: %s' % i_n)
    if co.co_freevars:
        lines.append('Free variables:')
        for i_n in enumerate(co.co_freevars):
            lines.append('%4d: %s' % i_n)
    if co.co_cellvars:
        lines.append('Cell variables:')
        for i_n in enumerate(co.co_cellvars):
            lines.append('%4d: %s' % i_n)
    return '\n'.join(lines)