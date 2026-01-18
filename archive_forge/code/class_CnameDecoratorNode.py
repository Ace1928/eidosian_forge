from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
class CnameDecoratorNode(StatNode):
    """
    This node is for the cname decorator in CythonUtilityCode:

        @cname('the_cname')
        cdef func(...):
            ...

    In case of a cdef class the cname specifies the objstruct_cname.

    node        the node to which the cname decorator is applied
    cname       the cname the node should get
    """
    child_attrs = ['node']

    def analyse_declarations(self, env):
        self.node.analyse_declarations(env)
        node = self.node
        if isinstance(node, CompilerDirectivesNode):
            node = node.body.stats[0]
        self.is_function = isinstance(node, FuncDefNode)
        is_struct_or_enum = isinstance(node, (CStructOrUnionDefNode, CEnumDefNode))
        e = node.entry
        if self.is_function:
            e.cname = self.cname
            e.func_cname = self.cname
            e.used = True
            if e.pyfunc_cname and '.' in e.pyfunc_cname:
                e.pyfunc_cname = self.mangle(e.pyfunc_cname)
        elif is_struct_or_enum:
            e.cname = e.type.cname = self.cname
        else:
            scope = node.scope
            e.cname = self.cname
            e.type.objstruct_cname = self.cname + '_obj'
            e.type.typeobj_cname = Naming.typeobj_prefix + self.cname
            e.type.typeptr_cname = self.cname + '_type'
            e.type.scope.namespace_cname = e.type.typeptr_cname
            e.as_variable.cname = e.type.typeptr_cname
            scope.scope_prefix = self.cname + '_'
            for name, entry in scope.entries.items():
                if entry.func_cname:
                    entry.func_cname = self.mangle(entry.cname)
                if entry.pyfunc_cname:
                    entry.pyfunc_cname = self.mangle(entry.pyfunc_cname)

    def mangle(self, cname):
        if '.' in cname:
            cname = cname.split('.')[-1]
        return '%s_%s' % (self.cname, cname)

    def analyse_expressions(self, env):
        self.node = self.node.analyse_expressions(env)
        return self

    def generate_function_definitions(self, env, code):
        """Ensure a prototype for every @cname method in the right place"""
        if self.is_function and env.is_c_class_scope:
            h_code = code.globalstate['utility_code_proto']
            if isinstance(self.node, DefNode):
                self.node.generate_function_header(h_code, with_pymethdef=False, proto_only=True)
            else:
                from . import ModuleNode
                entry = self.node.entry
                cname = entry.cname
                entry.cname = entry.func_cname
                ModuleNode.generate_cfunction_declaration(entry, env.global_scope(), h_code, definition=True)
                entry.cname = cname
        self.node.generate_function_definitions(env, code)

    def generate_execution_code(self, code):
        self.node.generate_execution_code(code)