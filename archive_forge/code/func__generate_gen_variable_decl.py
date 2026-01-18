import sys, os
import types
from . import model
from .error import VerificationError
def _generate_gen_variable_decl(self, tp, name):
    if isinstance(tp, model.ArrayType):
        if tp.length_is_unknown():
            prnt = self._prnt
            funcname = '_cffi_sizeof_%s' % (name,)
            self.export_symbols.append(funcname)
            prnt('size_t %s(void)' % funcname)
            prnt('{')
            prnt('  return sizeof(%s);' % (name,))
            prnt('}')
        tp_ptr = model.PointerType(tp.item)
        self._generate_gen_const(False, name, tp_ptr)
    else:
        tp_ptr = model.PointerType(tp)
        self._generate_gen_const(False, name, tp_ptr, category='var')