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
class UnicodeNode(ConstNode):
    is_string_literal = True
    bytes_value = None
    type = unicode_type

    def calculate_constant_result(self):
        self.constant_result = self.value

    def analyse_as_type(self, env):
        return _analyse_name_as_type(self.value, self.pos, env)

    def as_sliced_node(self, start, stop, step=None):
        if StringEncoding.string_contains_surrogates(self.value[:stop]):
            return None
        value = StringEncoding.EncodedString(self.value[start:stop:step])
        value.encoding = self.value.encoding
        if self.bytes_value is not None:
            bytes_value = StringEncoding.bytes_literal(self.bytes_value[start:stop:step], self.bytes_value.encoding)
        else:
            bytes_value = None
        return UnicodeNode(self.pos, value=value, bytes_value=bytes_value, constant_result=value)

    def coerce_to(self, dst_type, env):
        if dst_type is self.type:
            pass
        elif dst_type.is_unicode_char:
            if not self.can_coerce_to_char_literal():
                error(self.pos, 'Only single-character Unicode string literals or surrogate pairs can be coerced into Py_UCS4/Py_UNICODE.')
                return self
            int_value = ord(self.value)
            return IntNode(self.pos, type=dst_type, value=str(int_value), constant_result=int_value)
        elif not dst_type.is_pyobject:
            if dst_type.is_string and self.bytes_value is not None:
                return BytesNode(self.pos, value=self.bytes_value).coerce_to(dst_type, env)
            if dst_type.is_pyunicode_ptr:
                return UnicodeNode(self.pos, value=self.value, type=dst_type)
            error(self.pos, 'Unicode literals do not support coercion to C types other than Py_UNICODE/Py_UCS4 (for characters) or Py_UNICODE* (for strings).')
        elif dst_type not in (py_object_type, Builtin.basestring_type):
            self.check_for_coercion_error(dst_type, env, fail=True)
        return self

    def can_coerce_to_char_literal(self):
        return len(self.value) == 1

    def coerce_to_boolean(self, env):
        bool_value = bool(self.value)
        return BoolNode(self.pos, value=bool_value, constant_result=bool_value)

    def contains_surrogates(self):
        return StringEncoding.string_contains_surrogates(self.value)

    def generate_evaluation_code(self, code):
        if self.type.is_pyobject:
            if StringEncoding.string_contains_lone_surrogates(self.value):
                self.result_code = code.get_py_const(py_object_type, 'ustring')
                data_cname = code.get_string_const(StringEncoding.BytesLiteral(self.value.encode('unicode_escape')))
                const_code = code.get_cached_constants_writer(self.result_code)
                if const_code is None:
                    return
                const_code.mark_pos(self.pos)
                const_code.putln('%s = PyUnicode_DecodeUnicodeEscape(%s, sizeof(%s) - 1, NULL); %s' % (self.result_code, data_cname, data_cname, const_code.error_goto_if_null(self.result_code, self.pos)))
                const_code.put_error_if_neg(self.pos, '__Pyx_PyUnicode_READY(%s)' % self.result_code)
            else:
                self.result_code = code.get_py_string_const(self.value)
        else:
            self.result_code = code.get_pyunicode_ptr_const(self.value)

    def calculate_result_code(self):
        return self.result_code

    def compile_time_value(self, env):
        return self.value