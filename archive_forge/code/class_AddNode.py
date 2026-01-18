from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
class AddNode(NumBinopNode):

    def is_py_operation_types(self, type1, type2):
        if type1.is_string and type2.is_string or (type1.is_pyunicode_ptr and type2.is_pyunicode_ptr):
            return 1
        else:
            return NumBinopNode.is_py_operation_types(self, type1, type2)

    def infer_builtin_types_operation(self, type1, type2):
        string_types = (bytes_type, bytearray_type, str_type, basestring_type, unicode_type)
        if type1 in string_types and type2 in string_types:
            return string_types[max(string_types.index(type1), string_types.index(type2))]
        return None

    def compute_c_result_type(self, type1, type2):
        if (type1.is_ptr or type1.is_array) and (type2.is_int or type2.is_enum):
            return type1
        elif (type2.is_ptr or type2.is_array) and (type1.is_int or type1.is_enum):
            return type2
        else:
            return NumBinopNode.compute_c_result_type(self, type1, type2)

    def py_operation_function(self, code):
        type1, type2 = (self.operand1.type, self.operand2.type)
        func = None
        if type1 is unicode_type or type2 is unicode_type:
            if type1 in (unicode_type, str_type) and type2 in (unicode_type, str_type):
                is_unicode_concat = True
            elif isinstance(self.operand1, FormattedValueNode) or isinstance(self.operand2, FormattedValueNode):
                is_unicode_concat = True
            else:
                is_unicode_concat = False
            if is_unicode_concat:
                if self.inplace or self.operand1.is_temp:
                    code.globalstate.use_utility_code(UtilityCode.load_cached('UnicodeConcatInPlace', 'ObjectHandling.c'))
                func = '__Pyx_PyUnicode_Concat'
        elif type1 is str_type and type2 is str_type:
            code.globalstate.use_utility_code(UtilityCode.load_cached('StrConcatInPlace', 'ObjectHandling.c'))
            func = '__Pyx_PyStr_Concat'
        if func:
            if self.inplace or self.operand1.is_temp:
                func += 'InPlace'
            if self.operand1.may_be_none() or self.operand2.may_be_none():
                func += 'Safe'
            return func
        return super(AddNode, self).py_operation_function(code)