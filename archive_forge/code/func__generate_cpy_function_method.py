import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp
def _generate_cpy_function_method(self, tp, name):
    if tp.ellipsis:
        return
    numargs = len(tp.args)
    if numargs == 0:
        meth = 'METH_NOARGS'
    elif numargs == 1:
        meth = 'METH_O'
    else:
        meth = 'METH_VARARGS'
    self._prnt('  {"%s", _cffi_f_%s, %s, NULL},' % (name, name, meth))