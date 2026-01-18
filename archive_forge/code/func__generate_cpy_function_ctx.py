import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_function_ctx(self, tp, name):
    if tp.ellipsis and (not self.target_is_python):
        self._generate_cpy_constant_ctx(tp, name)
        return
    type_index = self._typesdict[tp.as_raw_function()]
    numargs = len(tp.args)
    if self.target_is_python:
        meth_kind = OP_DLOPEN_FUNC
    elif numargs == 0:
        meth_kind = OP_CPYTHON_BLTN_N
    elif numargs == 1:
        meth_kind = OP_CPYTHON_BLTN_O
    else:
        meth_kind = OP_CPYTHON_BLTN_V
    self._lsts['global'].append(GlobalExpr(name, '_cffi_f_%s' % name, CffiOp(meth_kind, type_index), size='_cffi_d_%s' % name))