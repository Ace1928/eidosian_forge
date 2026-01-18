import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _add_missing_struct_unions(self):
    lst = list(self._struct_unions.items())
    lst.sort(key=lambda tp_order: tp_order[1])
    for tp, order in lst:
        if tp not in self._seen_struct_unions:
            if tp.partial:
                raise NotImplementedError('internal inconsistency: %r is partial but was not seen at this point' % (tp,))
            if tp.name.startswith('$') and tp.name[1:].isdigit():
                approxname = tp.name[1:]
            elif tp.name == '_IO_FILE' and tp.forcename == 'FILE':
                approxname = 'FILE'
                self._typedef_ctx(tp, 'FILE')
            else:
                raise NotImplementedError('internal inconsistency: %r' % (tp,))
            self._struct_ctx(tp, None, approxname)