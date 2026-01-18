import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_extern_python_ctx(self, tp, name):
    if self.target_is_python:
        raise VerificationError('cannot use \'extern "Python"\' in the ABI mode')
    if tp.ellipsis:
        raise NotImplementedError('a vararg function is extern "Python"')
    type_index = self._typesdict[tp]
    type_op = CffiOp(OP_EXTERN_PYTHON, type_index)
    self._lsts['global'].append(GlobalExpr(name, '&_cffi_externpy__%s' % name, type_op, name))