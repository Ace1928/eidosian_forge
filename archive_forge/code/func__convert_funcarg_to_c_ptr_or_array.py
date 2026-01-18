import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _convert_funcarg_to_c_ptr_or_array(self, tp, fromvar, tovar, errcode):
    self._prnt('  datasize = _cffi_prepare_pointer_call_argument(')
    self._prnt('      _cffi_type(%d), %s, (char **)&%s);' % (self._gettypenum(tp), fromvar, tovar))
    self._prnt('  if (datasize != 0) {')
    self._prnt('    %s = ((size_t)datasize) <= 640 ? (%s)alloca((size_t)datasize) : NULL;' % (tovar, tp.get_c_name('')))
    self._prnt('    if (_cffi_convert_array_argument(_cffi_type(%d), %s, (char **)&%s,' % (self._gettypenum(tp), fromvar, tovar))
    self._prnt('            datasize, &large_args_free) < 0)')
    self._prnt('      %s;' % errcode)
    self._prnt('  }')