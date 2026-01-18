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
class _AssignmentExpressionChecker(TreeVisitor):
    """
    Enforces rules on AssignmentExpressions within generator expressions and comprehensions
    """

    def __init__(self, loop_node, scope_is_class):
        super(_AssignmentExpressionChecker, self).__init__()
        target_name_finder = _AssignmentExpressionTargetNameFinder()
        target_name_finder.visit(loop_node)
        self.target_names_dict = target_name_finder.target_names
        self.in_iterator = False
        self.in_nested_generator = False
        self.scope_is_class = scope_is_class
        self.current_target_names = ()
        self.all_target_names = set()
        for names in self.target_names_dict.values():
            self.all_target_names.update(names)

    def _reset_state(self):
        old_state = (self.in_iterator, self.in_nested_generator, self.scope_is_class, self.all_target_names, self.current_target_names)
        self.in_nested_generator = False
        self.scope_is_class = False
        self.current_target_names = ()
        self.all_target_names = set()
        return old_state

    def _set_state(self, old_state):
        self.in_iterator, self.in_nested_generator, self.scope_is_class, self.all_target_names, self.current_target_names = old_state

    @classmethod
    def do_checks(cls, loop_node, scope_is_class):
        checker = cls(loop_node, scope_is_class)
        checker.visit(loop_node)

    def visit_ForInStatNode(self, node):
        if self.in_nested_generator:
            self.visitchildren(node)
            return
        current_target_names = self.current_target_names
        target_name = self.target_names_dict.get(node, None)
        if target_name:
            self.current_target_names += target_name
        self.in_iterator = True
        self.visit(node.iterator)
        self.in_iterator = False
        self.visitchildren(node, exclude=('iterator',))
        self.current_target_names = current_target_names

    def visit_AssignmentExpressionNode(self, node):
        if self.in_iterator:
            error(node.pos, 'assignment expression cannot be used in a comprehension iterable expression')
        if self.scope_is_class:
            error(node.pos, 'assignment expression within a comprehension cannot be used in a class body')
        if node.target_name in self.current_target_names:
            error(node.pos, "assignment expression cannot rebind comprehension iteration variable '%s'" % node.target_name)
        elif node.target_name in self.all_target_names:
            error(node.pos, "comprehension inner loop cannot rebind assignment expression target '%s'" % node.target_name)

    def visit_LambdaNode(self, node):
        old_state = self._reset_state()
        self.visit(node.result_expr)
        self._set_state(old_state)

    def visit_ComprehensionNode(self, node):
        in_nested_generator = self.in_nested_generator
        self.in_nested_generator = True
        self.visitchildren(node)
        self.in_nested_generator = in_nested_generator

    def visit_GeneratorExpressionNode(self, node):
        in_nested_generator = self.in_nested_generator
        self.in_nested_generator = True
        self.visit(node.loop)
        self.in_nested_generator = in_nested_generator

    def visit_Node(self, node):
        self.visitchildren(node)