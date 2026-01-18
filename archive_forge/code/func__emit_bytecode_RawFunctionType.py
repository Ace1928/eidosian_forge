import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _emit_bytecode_RawFunctionType(self, tp, index):
    self.cffi_types[index] = CffiOp(OP_FUNCTION, self._typesdict[tp.result])
    index += 1
    for tp1 in tp.args:
        realindex = self._typesdict[tp1]
        if index != realindex:
            if isinstance(tp1, model.PrimitiveType):
                self._emit_bytecode_PrimitiveType(tp1, index)
            else:
                self.cffi_types[index] = CffiOp(OP_NOOP, realindex)
        index += 1
    flags = int(tp.ellipsis)
    if tp.abi is not None:
        if tp.abi == '__stdcall':
            flags |= 2
        else:
            raise NotImplementedError('abi=%r' % (tp.abi,))
    self.cffi_types[index] = CffiOp(OP_FUNCTION_END, flags)