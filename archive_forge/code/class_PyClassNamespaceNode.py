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
class PyClassNamespaceNode(ExprNode, ModuleNameMixin):
    subexprs = ['doc']

    def analyse_types(self, env):
        if self.doc:
            self.doc = self.doc.analyse_types(env).coerce_to_pyobject(env)
        self.type = py_object_type
        self.is_temp = 1
        return self

    def may_be_none(self):
        return True

    def generate_result_code(self, code):
        cname = code.intern_identifier(self.name)
        py_mod_name = self.get_py_mod_name(code)
        qualname = self.get_py_qualified_name(code)
        class_def_node = self.class_def_node
        null = '(PyObject *) NULL'
        doc_code = self.doc.result() if self.doc else null
        mkw = class_def_node.mkw.py_result() if class_def_node.mkw else null
        metaclass = class_def_node.metaclass.py_result() if class_def_node.metaclass else null
        code.putln('%s = __Pyx_Py3MetaclassPrepare(%s, %s, %s, %s, %s, %s, %s); %s' % (self.result(), metaclass, class_def_node.bases.result(), cname, qualname, mkw, py_mod_name, doc_code, code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)