import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_enum_decl(self, tp, name=None):
    for enumerator in tp.enumerators:
        self._generate_cpy_const(True, enumerator)