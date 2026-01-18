import contextlib
import functools
from llvmlite.ir import instructions, types, values
def _castop(opname, cls=instructions.CastInstr):

    def wrap(fn):

        @functools.wraps(fn)
        def wrapped(self, val, typ, name=''):
            if val.type == typ:
                return val
            instr = cls(self.block, opname, val, typ, name)
            self._insert(instr)
            return instr
        return wrapped
    return wrap