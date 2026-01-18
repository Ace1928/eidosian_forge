from __future__ import absolute_import
from .StringEncoding import EncodedString
from .Symtab import BuiltinScope, StructOrUnionScope, ModuleScope, Entry
from .Code import UtilityCode, TempitaUtilityCode
from .TypeSlots import Signature
from . import PyrexTypes
class BuiltinMethod(_BuiltinOverride):

    def declare_in_type(self, self_type):
        method_type, sig = (self.func_type, self.sig)
        if method_type is None:
            self_arg = PyrexTypes.CFuncTypeArg('', self_type, None)
            self_arg.not_none = True
            self_arg.accept_builtin_subtypes = True
            method_type = self.build_func_type(sig, self_arg)
        self_type.scope.declare_builtin_cfunction(self.py_name, method_type, self.cname, utility_code=self.utility_code)