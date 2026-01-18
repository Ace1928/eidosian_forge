import contextlib
import functools
from llvmlite.ir import instructions, types, values
def _unop(opname, cls=instructions.Instruction):

    def wrap(fn):

        @functools.wraps(fn)
        def wrapped(self, arg, name='', flags=()):
            instr = cls(self.block, arg.type, opname, [arg], name, flags)
            self._insert(instr)
            return instr
        return wrapped
    return wrap