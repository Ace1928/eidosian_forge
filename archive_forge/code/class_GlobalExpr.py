import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
class GlobalExpr:

    def __init__(self, name, address, type_op, size=0, check_value=0):
        self.name = name
        self.address = address
        self.type_op = type_op
        self.size = size
        self.check_value = check_value

    def as_c_expr(self):
        return '  { "%s", (void *)%s, %s, (void *)%s },' % (self.name, self.address, self.type_op.as_c_expr(), self.size)

    def as_python_expr(self):
        return "b'%s%s',%d" % (self.type_op.as_python_bytes(), self.name, self.check_value)