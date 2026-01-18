import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _emit_bytecode_StructType(self, tp, index):
    struct_index = self._struct_unions[tp]
    self.cffi_types[index] = CffiOp(OP_STRUCT_UNION, struct_index)