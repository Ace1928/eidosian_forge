import contextlib
import functools
from llvmlite.ir import instructions, types, values
def _icmp(self, prefix, cmpop, lhs, rhs, name):
    try:
        op = _CMP_MAP[cmpop]
    except KeyError:
        raise ValueError('invalid comparison %r for icmp' % (cmpop,))
    if cmpop not in ('==', '!='):
        op = prefix + op
    instr = instructions.ICMPInstr(self.block, op, lhs, rhs, name=name)
    self._insert(instr)
    return instr