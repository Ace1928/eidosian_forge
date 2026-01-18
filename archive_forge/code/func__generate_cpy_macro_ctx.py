import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_macro_ctx(self, tp, name):
    if tp == '...':
        if self.target_is_python:
            raise VerificationError("cannot use the syntax '...' in '#define %s ...' when using the ABI mode" % (name,))
        check_value = None
    else:
        check_value = tp
    type_op = CffiOp(OP_CONSTANT_INT, -1)
    self._lsts['global'].append(GlobalExpr(name, '_cffi_const_%s' % name, type_op, check_value=check_value))