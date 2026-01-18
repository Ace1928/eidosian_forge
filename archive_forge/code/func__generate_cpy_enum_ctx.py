import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_enum_ctx(self, tp, name):
    self._enum_ctx(tp, tp._get_c_name())