import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_anonymous_ctx(self, tp, name):
    if isinstance(tp, model.EnumType):
        self._enum_ctx(tp, name)
    else:
        self._struct_ctx(tp, name, 'typedef_' + name)