import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_struct_ctx(self, tp, name):
    self._struct_ctx(tp, *self._struct_names(tp))