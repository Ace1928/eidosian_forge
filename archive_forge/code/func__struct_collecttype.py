import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _struct_collecttype(self, tp):
    self._do_collect_type(tp)
    if self.target_is_python:
        for fldtype in tp.anonymous_struct_fields():
            self._struct_collecttype(fldtype)