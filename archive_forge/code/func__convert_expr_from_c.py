import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _convert_expr_from_c(self, tp, var, context):
    if isinstance(tp, model.BasePrimitiveType):
        if tp.is_integer_type() and tp.name != '_Bool':
            return '_cffi_from_c_int(%s, %s)' % (var, tp.name)
        elif isinstance(tp, model.UnknownFloatType):
            return '_cffi_from_c_double(%s)' % (var,)
        elif tp.name != 'long double' and (not tp.is_complex_type()):
            cname = tp.name.replace(' ', '_')
            if cname in ('char16_t', 'char32_t'):
                self.needs_version(VERSION_CHAR16CHAR32)
            return '_cffi_from_c_%s(%s)' % (cname, var)
        else:
            return '_cffi_from_c_deref((char *)&%s, _cffi_type(%d))' % (var, self._gettypenum(tp))
    elif isinstance(tp, (model.PointerType, model.FunctionPtrType)):
        return '_cffi_from_c_pointer((char *)%s, _cffi_type(%d))' % (var, self._gettypenum(tp))
    elif isinstance(tp, model.ArrayType):
        return '_cffi_from_c_pointer((char *)%s, _cffi_type(%d))' % (var, self._gettypenum(model.PointerType(tp.item)))
    elif isinstance(tp, model.StructOrUnion):
        if tp.fldnames is None:
            raise TypeError("'%s' is used as %s, but is opaque" % (tp._get_c_name(), context))
        return '_cffi_from_c_struct((char *)&%s, _cffi_type(%d))' % (var, self._gettypenum(tp))
    elif isinstance(tp, model.EnumType):
        return '_cffi_from_c_deref((char *)&%s, _cffi_type(%d))' % (var, self._gettypenum(tp))
    else:
        raise NotImplementedError(tp)