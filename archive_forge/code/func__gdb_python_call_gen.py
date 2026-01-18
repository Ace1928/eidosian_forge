import numpy as np
from numba.core.typing.typeof import typeof
from numba.core.typing.asnumbatype import as_numba_type
def _gdb_python_call_gen(func_name, *args):
    import numba
    fn = getattr(numba, func_name)
    argstr = ','.join(['"%s"' for _ in args]) % args
    defn = 'def _gdb_func_injection():\n\t%s(%s)\n\n    ' % (func_name, argstr)
    l = {}
    exec(defn, {func_name: fn}, l)
    return numba.njit(l['_gdb_func_injection'])