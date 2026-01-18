from __future__ import absolute_import
from .StringEncoding import EncodedString
from .Symtab import BuiltinScope, StructOrUnionScope, ModuleScope, Entry
from .Code import UtilityCode, TempitaUtilityCode
from .TypeSlots import Signature
from . import PyrexTypes
class BuiltinProperty(object):

    def __init__(self, py_name, property_type, call_cname, exception_value=None, exception_check=None, utility_code=None):
        self.py_name = py_name
        self.property_type = property_type
        self.call_cname = call_cname
        self.utility_code = utility_code
        self.exception_value = exception_value
        self.exception_check = exception_check

    def declare_in_type(self, self_type):
        self_type.scope.declare_cproperty(self.py_name, self.property_type, self.call_cname, exception_value=self.exception_value, exception_check=self.exception_check, utility_code=self.utility_code)