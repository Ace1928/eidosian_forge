import contextlib
import functools
from llvmlite.ir import instructions, types, values
def _triop_intrinsic(opname):

    def wrap(fn):

        @functools.wraps(fn)
        def wrapped(self, a, b, c, name=''):
            if a.type != b.type or b.type != c.type:
                raise TypeError('expected types to be the same, got %s, %s, %s' % (a.type, b.type, c.type))
            elif not isinstance(a.type, (types.HalfType, types.FloatType, types.DoubleType)):
                raise TypeError('expected an floating point type, got %s' % a.type)
            fn = self.module.declare_intrinsic(opname, [a.type, b.type, c.type])
            return self.call(fn, [a, b, c], name)
        return wrapped
    return wrap