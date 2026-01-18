import sys
from llvmlite import ir
import llvmlite.binding as ll
from numba.core import utils, intrinsics
from numba import _helperlib
def _do_install(self, context):
    is32bit = utils.MACHINE_BITS == 32
    c_helpers = _helperlib.c_helpers
    if sys.platform.startswith('win32') and is32bit:
        ftol = _get_msvcrt_symbol('_ftol')
        _add_missing_symbol('_ftol2', ftol)
    elif sys.platform.startswith('linux') and is32bit:
        _add_missing_symbol('__fixunsdfdi', c_helpers['fptoui'])
        _add_missing_symbol('__fixunssfdi', c_helpers['fptouif'])
    if is32bit:
        self._multi3_lib = compile_multi3(context)
        ptr = self._multi3_lib.get_pointer_to_function('multi3')
        assert ptr
        _add_missing_symbol('__multi3', ptr)
    for fname in intrinsics.INTR_MATH:
        ll.add_symbol(fname, c_helpers[fname])