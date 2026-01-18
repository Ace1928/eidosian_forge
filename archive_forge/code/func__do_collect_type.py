import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _do_collect_type(self, tp):
    if not isinstance(tp, model.BaseTypeByIdentity):
        if isinstance(tp, tuple):
            for x in tp:
                self._do_collect_type(x)
        return
    if tp not in self._typesdict:
        self._typesdict[tp] = None
        if isinstance(tp, model.FunctionPtrType):
            self._do_collect_type(tp.as_raw_function())
        elif isinstance(tp, model.StructOrUnion):
            if tp.fldtypes is not None and tp not in self.ffi._parser._included_declarations:
                for name1, tp1, _, _ in self._enum_fields(tp):
                    self._do_collect_type(self._field_type(tp, name1, tp1))
        else:
            for _, x in tp._get_items():
                self._do_collect_type(x)