from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
class CalculateQualifiedNamesTransform(EnvTransform):
    """
    Calculate and store the '__qualname__' and the global
    module name on some nodes.
    """
    needs_qualname_assignment = False
    needs_module_assignment = False

    def visit_ModuleNode(self, node):
        self.module_name = self.global_scope().qualified_name
        self.qualified_name = []
        _super = super(CalculateQualifiedNamesTransform, self)
        self._super_visit_FuncDefNode = _super.visit_FuncDefNode
        self._super_visit_ClassDefNode = _super.visit_ClassDefNode
        self.visitchildren(node)
        return node

    def _set_qualname(self, node, name=None):
        if name:
            qualname = self.qualified_name[:]
            qualname.append(name)
        else:
            qualname = self.qualified_name
        node.qualname = EncodedString('.'.join(qualname))
        node.module_name = self.module_name

    def _append_entry(self, entry):
        if entry.is_pyglobal and (not entry.is_pyclass_attr):
            self.qualified_name = [entry.name]
        else:
            self.qualified_name.append(entry.name)

    def visit_ClassNode(self, node):
        self._set_qualname(node, node.name)
        self.visitchildren(node)
        return node

    def visit_PyClassNamespaceNode(self, node):
        self._set_qualname(node)
        self.visitchildren(node)
        return node

    def visit_PyCFunctionNode(self, node):
        orig_qualified_name = self.qualified_name[:]
        if node.def_node.is_wrapper and self.qualified_name and (self.qualified_name[-1] == '<locals>'):
            self.qualified_name.pop()
            self._set_qualname(node)
        else:
            self._set_qualname(node, node.def_node.name)
        self.visitchildren(node)
        self.qualified_name = orig_qualified_name
        return node

    def visit_DefNode(self, node):
        if node.is_wrapper and self.qualified_name:
            assert self.qualified_name[-1] == '<locals>', self.qualified_name
            orig_qualified_name = self.qualified_name[:]
            self.qualified_name.pop()
            self._set_qualname(node)
            self._super_visit_FuncDefNode(node)
            self.qualified_name = orig_qualified_name
        else:
            self._set_qualname(node, node.name)
            self.visit_FuncDefNode(node)
        return node

    def visit_FuncDefNode(self, node):
        orig_qualified_name = self.qualified_name[:]
        if getattr(node, 'name', None) == '<lambda>':
            self.qualified_name.append('<lambda>')
        else:
            self._append_entry(node.entry)
        self.qualified_name.append('<locals>')
        self._super_visit_FuncDefNode(node)
        self.qualified_name = orig_qualified_name
        return node

    def generate_assignment(self, node, name, value):
        entry = node.scope.lookup_here(name)
        lhs = ExprNodes.NameNode(node.pos, name=EncodedString(name), entry=entry)
        rhs = ExprNodes.StringNode(node.pos, value=value.as_utf8_string(), unicode_value=value)
        node.body.stats.insert(0, Nodes.SingleAssignmentNode(node.pos, lhs=lhs, rhs=rhs).analyse_expressions(self.current_env()))

    def visit_ClassDefNode(self, node):
        orig_needs_qualname_assignment = self.needs_qualname_assignment
        self.needs_qualname_assignment = False
        orig_needs_module_assignment = self.needs_module_assignment
        self.needs_module_assignment = False
        orig_qualified_name = self.qualified_name[:]
        entry = getattr(node, 'entry', None) or self.current_env().lookup_here(node.target.name)
        self._append_entry(entry)
        self._super_visit_ClassDefNode(node)
        if self.needs_qualname_assignment:
            self.generate_assignment(node, '__qualname__', EncodedString('.'.join(self.qualified_name)))
        if self.needs_module_assignment:
            self.generate_assignment(node, '__module__', EncodedString(self.module_name))
        self.qualified_name = orig_qualified_name
        self.needs_qualname_assignment = orig_needs_qualname_assignment
        self.needs_module_assignment = orig_needs_module_assignment
        return node

    def visit_NameNode(self, node):
        scope = self.current_env()
        if scope.is_c_class_scope:
            if node.name == '__qualname__':
                self.needs_qualname_assignment = True
            elif node.name == '__module__':
                self.needs_module_assignment = True
        return node