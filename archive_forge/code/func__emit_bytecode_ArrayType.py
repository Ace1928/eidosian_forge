import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _emit_bytecode_ArrayType(self, tp, index):
    item_index = self._typesdict[tp.item]
    if tp.length is None:
        self.cffi_types[index] = CffiOp(OP_OPEN_ARRAY, item_index)
    elif tp.length == '...':
        raise VerificationError("type %s badly placed: the '...' array length can only be used on global arrays or on fields of structures" % (str(tp).replace('/*...*/', '...'),))
    else:
        assert self.cffi_types[index + 1] == 'LEN'
        self.cffi_types[index] = CffiOp(OP_ARRAY, item_index)
        self.cffi_types[index + 1] = CffiOp(None, str(tp.length))