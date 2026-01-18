import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_typedef_ctx(self, tp, name):
    tp = self._typedef_type(tp, name)
    self._typedef_ctx(tp, name)
    if getattr(tp, 'origin', None) == 'unknown_type':
        self._struct_ctx(tp, tp.name, approxname=None)
    elif isinstance(tp, model.NamedPointerType):
        self._struct_ctx(tp.totype, tp.totype.name, approxname=tp.name, named_ptr=tp)