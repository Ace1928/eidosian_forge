import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_variable_decl(self, tp, name):
    prnt = self._prnt
    tp = self._global_type(tp, name)
    if isinstance(tp, model.ArrayType) and tp.length is None:
        tp = tp.item
        ampersand = ''
    else:
        ampersand = '&'
    decl = '*_cffi_var_%s(void)' % (name,)
    prnt('static ' + tp.get_c_name(decl, quals=self._current_quals))
    prnt('{')
    prnt('  return %s(%s);' % (ampersand, name))
    prnt('}')
    prnt()