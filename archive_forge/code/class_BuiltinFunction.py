from __future__ import absolute_import
from .StringEncoding import EncodedString
from .Symtab import BuiltinScope, StructOrUnionScope, ModuleScope, Entry
from .Code import UtilityCode, TempitaUtilityCode
from .TypeSlots import Signature
from . import PyrexTypes
class BuiltinFunction(_BuiltinOverride):

    def declare_in_scope(self, scope):
        func_type, sig = (self.func_type, self.sig)
        if func_type is None:
            func_type = self.build_func_type(sig)
        scope.declare_builtin_cfunction(self.py_name, func_type, self.cname, self.py_equiv, self.utility_code)