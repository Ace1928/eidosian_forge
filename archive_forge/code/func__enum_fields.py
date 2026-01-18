import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _enum_fields(self, tp):
    expand_anonymous_struct_union = not self.target_is_python
    return tp.enumfields(expand_anonymous_struct_union)