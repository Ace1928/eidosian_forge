import contextlib
import functools
from llvmlite.ir import instructions, types, values
def _uniop(opname, cls=instructions.Instruction):

    def wrap(fn):

        @functools.wraps(fn)
        def wrapped(self, operand, name=''):
            instr = cls(self.block, operand.type, opname, [operand], name)
            self._insert(instr)
            return instr
        return wrapped
    return wrap