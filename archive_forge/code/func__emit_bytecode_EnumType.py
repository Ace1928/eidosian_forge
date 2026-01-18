import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _emit_bytecode_EnumType(self, tp, index):
    enum_index = self._enums[tp]
    self.cffi_types[index] = CffiOp(OP_ENUM, enum_index)