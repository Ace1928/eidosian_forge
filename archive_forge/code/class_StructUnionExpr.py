import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
class StructUnionExpr:

    def __init__(self, name, type_index, flags, size, alignment, comment, first_field_index, c_fields):
        self.name = name
        self.type_index = type_index
        self.flags = flags
        self.size = size
        self.alignment = alignment
        self.comment = comment
        self.first_field_index = first_field_index
        self.c_fields = c_fields

    def as_c_expr(self):
        return '  { "%s", %d, %s,' % (self.name, self.type_index, self.flags) + '\n    %s, %s, ' % (self.size, self.alignment) + '%d, %d ' % (self.first_field_index, len(self.c_fields)) + ('/* %s */ ' % self.comment if self.comment else '') + '},'

    def as_python_expr(self):
        flags = eval(self.flags, G_FLAGS)
        fields_expr = [c_field.as_field_python_expr() for c_field in self.c_fields]
        return "(b'%s%s%s',%s)" % (format_four_bytes(self.type_index), format_four_bytes(flags), self.name, ','.join(fields_expr))