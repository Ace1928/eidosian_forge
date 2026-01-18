import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _check_not_opaque(self, tp, location):
    while isinstance(tp, model.ArrayType):
        tp = tp.item
    if isinstance(tp, model.StructOrUnion) and tp.fldtypes is None:
        raise TypeError('%s is of an opaque type (not declared in cdef())' % location)