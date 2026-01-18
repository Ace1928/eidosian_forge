import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _field_type(self, tp_struct, field_name, tp_field):
    if isinstance(tp_field, model.ArrayType):
        actual_length = tp_field.length
        if actual_length == '...':
            ptr_struct_name = tp_struct.get_c_name('*')
            actual_length = '_cffi_array_len(((%s)0)->%s)' % (ptr_struct_name, field_name)
        tp_item = self._field_type(tp_struct, '%s[0]' % field_name, tp_field.item)
        tp_field = model.ArrayType(tp_item, actual_length)
    return tp_field