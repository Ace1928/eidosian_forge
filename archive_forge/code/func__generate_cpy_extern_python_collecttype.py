import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_extern_python_collecttype(self, tp, name):
    assert isinstance(tp, model.FunctionPtrType)
    self._do_collect_type(tp)