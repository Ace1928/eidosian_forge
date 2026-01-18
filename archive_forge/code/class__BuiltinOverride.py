from __future__ import absolute_import
from .StringEncoding import EncodedString
from .Symtab import BuiltinScope, StructOrUnionScope, ModuleScope, Entry
from .Code import UtilityCode, TempitaUtilityCode
from .TypeSlots import Signature
from . import PyrexTypes
class _BuiltinOverride(object):

    def __init__(self, py_name, args, ret_type, cname, py_equiv='*', utility_code=None, sig=None, func_type=None, is_strict_signature=False, builtin_return_type=None, nogil=None):
        self.py_name, self.cname, self.py_equiv = (py_name, cname, py_equiv)
        self.args, self.ret_type = (args, ret_type)
        self.func_type, self.sig = (func_type, sig)
        self.builtin_return_type = builtin_return_type
        self.is_strict_signature = is_strict_signature
        self.utility_code = utility_code
        self.nogil = nogil

    def build_func_type(self, sig=None, self_arg=None):
        if sig is None:
            sig = Signature(self.args, self.ret_type, nogil=self.nogil)
            sig.exception_check = False
        func_type = sig.function_type(self_arg)
        if self.is_strict_signature:
            func_type.is_strict_signature = True
        if self.builtin_return_type:
            func_type.return_type = builtin_types[self.builtin_return_type]
        return func_type