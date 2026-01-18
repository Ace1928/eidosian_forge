import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_variable_collecttype(self, tp, name):
    self._do_collect_type(self._global_type(tp, name))