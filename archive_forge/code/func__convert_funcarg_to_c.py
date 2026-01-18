import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _convert_funcarg_to_c(self, tp, fromvar, tovar, errcode):
    extraarg = ''
    if isinstance(tp, model.BasePrimitiveType) and (not tp.is_complex_type()):
        if tp.is_integer_type() and tp.name != '_Bool':
            converter = '_cffi_to_c_int'
            extraarg = ', %s' % tp.name
        elif isinstance(tp, model.UnknownFloatType):
            converter = '(%s)_cffi_to_c_double' % (tp.get_c_name(''),)
        else:
            cname = tp.get_c_name('')
            converter = '(%s)_cffi_to_c_%s' % (cname, tp.name.replace(' ', '_'))
            if cname in ('char16_t', 'char32_t'):
                self.needs_version(VERSION_CHAR16CHAR32)
        errvalue = '-1'
    elif isinstance(tp, model.PointerType):
        self._convert_funcarg_to_c_ptr_or_array(tp, fromvar, tovar, errcode)
        return
    elif isinstance(tp, model.StructOrUnionOrEnum) or isinstance(tp, model.BasePrimitiveType):
        self._prnt('  if (_cffi_to_c((char *)&%s, _cffi_type(%d), %s) < 0)' % (tovar, self._gettypenum(tp), fromvar))
        self._prnt('    %s;' % errcode)
        return
    elif isinstance(tp, model.FunctionPtrType):
        converter = '(%s)_cffi_to_c_pointer' % tp.get_c_name('')
        extraarg = ', _cffi_type(%d)' % self._gettypenum(tp)
        errvalue = 'NULL'
    else:
        raise NotImplementedError(tp)
    self._prnt('  %s = %s(%s%s);' % (tovar, converter, fromvar, extraarg))
    self._prnt('  if (%s == (%s)%s && PyErr_Occurred())' % (tovar, tp.get_c_name(''), errvalue))
    self._prnt('    %s;' % errcode)