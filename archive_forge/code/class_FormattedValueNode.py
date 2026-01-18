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
class FormattedValueNode(ExprNode):
    subexprs = ['value', 'format_spec']
    type = unicode_type
    is_temp = True
    c_format_spec = None
    gil_message = 'String formatting'
    find_conversion_func = {'s': 'PyObject_Unicode', 'r': 'PyObject_Repr', 'a': 'PyObject_ASCII', 'd': '__Pyx_PyNumber_IntOrLong'}.get

    def may_be_none(self):
        return False

    def analyse_types(self, env):
        self.value = self.value.analyse_types(env)
        if not self.format_spec or self.format_spec.is_string_literal:
            c_format_spec = self.format_spec.value if self.format_spec else self.value.type.default_format_spec
            if self.value.type.can_coerce_to_pystring(env, format_spec=c_format_spec):
                self.c_format_spec = c_format_spec
        if self.format_spec:
            self.format_spec = self.format_spec.analyse_types(env).coerce_to_pyobject(env)
        if self.c_format_spec is None:
            self.value = self.value.coerce_to_pyobject(env)
            if not self.format_spec and (not self.conversion_char or self.conversion_char == 's'):
                if self.value.type is unicode_type and (not self.value.may_be_none()):
                    return self.value
        return self

    def generate_result_code(self, code):
        if self.c_format_spec is not None and (not self.value.type.is_pyobject):
            convert_func_call = self.value.type.convert_to_pystring(self.value.result(), code, self.c_format_spec)
            code.putln('%s = %s; %s' % (self.result(), convert_func_call, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
            return
        value_result = self.value.py_result()
        value_is_unicode = self.value.type is unicode_type and (not self.value.may_be_none())
        if self.format_spec:
            format_func = '__Pyx_PyObject_Format'
            format_spec = self.format_spec.py_result()
        else:
            format_func = '__Pyx_PyObject_FormatSimple'
            format_spec = Naming.empty_unicode
        conversion_char = self.conversion_char
        if conversion_char == 's' and value_is_unicode:
            conversion_char = None
        if conversion_char:
            fn = self.find_conversion_func(conversion_char)
            assert fn is not None, "invalid conversion character found: '%s'" % conversion_char
            value_result = '%s(%s)' % (fn, value_result)
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectFormatAndDecref', 'StringTools.c'))
            format_func += 'AndDecref'
        elif self.format_spec:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectFormat', 'StringTools.c'))
        else:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectFormatSimple', 'StringTools.c'))
        code.putln('%s = %s(%s, %s); %s' % (self.result(), format_func, value_result, format_spec, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)