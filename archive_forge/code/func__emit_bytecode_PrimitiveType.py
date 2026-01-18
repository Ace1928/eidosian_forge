import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _emit_bytecode_PrimitiveType(self, tp, index):
    prim_index = PRIMITIVE_TO_INDEX[tp.name]
    self.cffi_types[index] = CffiOp(OP_PRIMITIVE, prim_index)