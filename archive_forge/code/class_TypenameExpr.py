import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
class TypenameExpr:

    def __init__(self, name, type_index):
        self.name = name
        self.type_index = type_index

    def as_c_expr(self):
        return '  { "%s", %d },' % (self.name, self.type_index)

    def as_python_expr(self):
        return "b'%s%s'" % (format_four_bytes(self.type_index), self.name)