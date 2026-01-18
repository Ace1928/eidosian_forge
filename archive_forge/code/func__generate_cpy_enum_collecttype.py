import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_enum_collecttype(self, tp, name):
    self._do_collect_type(tp)