from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
class FinalOptimizePhase(Visitor.EnvTransform, Visitor.NodeRefCleanupMixin):
    """
    This visitor handles several commuting optimizations, and is run
    just before the C code generation phase.

    The optimizations currently implemented in this class are:
        - eliminate None assignment and refcounting for first assignment.
        - isinstance -> typecheck for cdef types
        - eliminate checks for None and/or types that became redundant after tree changes
        - eliminate useless string formatting steps
        - inject branch hints for unlikely if-cases that only raise exceptions
        - replace Python function calls that look like method calls by a faster PyMethodCallNode
    """
    in_loop = False

    def visit_SingleAssignmentNode(self, node):
        """Avoid redundant initialisation of local variables before their
        first assignment.
        """
        self.visitchildren(node)
        if node.first:
            lhs = node.lhs
            lhs.lhs_of_first_assignment = True
        return node

    def visit_SimpleCallNode(self, node):
        """
        Replace generic calls to isinstance(x, type) by a more efficient type check.
        Replace likely Python method calls by a specialised PyMethodCallNode.
        """
        self.visitchildren(node)
        function = node.function
        if function.type.is_cfunction and function.is_name:
            if function.name == 'isinstance' and len(node.args) == 2:
                type_arg = node.args[1]
                if type_arg.type.is_builtin_type and type_arg.type.name == 'type':
                    cython_scope = self.context.cython_scope
                    function.entry = cython_scope.lookup('PyObject_TypeCheck')
                    function.type = function.entry.type
                    PyTypeObjectPtr = PyrexTypes.CPtrType(cython_scope.lookup('PyTypeObject').type)
                    node.args[1] = ExprNodes.CastNode(node.args[1], PyTypeObjectPtr)
        elif node.is_temp and function.type.is_pyobject and self.current_directives.get('optimize.unpack_method_calls_in_pyinit' if not self.in_loop and self.current_env().is_module_scope else 'optimize.unpack_method_calls'):
            if isinstance(node.arg_tuple, ExprNodes.TupleNode) and (not (node.arg_tuple.mult_factor or (node.arg_tuple.is_literal and len(node.arg_tuple.args) > 1))):
                may_be_a_method = True
                if function.type is Builtin.type_type:
                    may_be_a_method = False
                elif function.is_attribute:
                    if function.entry and function.entry.type.is_cfunction:
                        may_be_a_method = False
                elif function.is_name:
                    entry = function.entry
                    if entry.is_builtin or entry.type.is_cfunction:
                        may_be_a_method = False
                    elif entry.cf_assignments:
                        non_method_nodes = (ExprNodes.PyCFunctionNode, ExprNodes.ClassNode, ExprNodes.Py3ClassNode)
                        may_be_a_method = any((assignment.rhs and (not isinstance(assignment.rhs, non_method_nodes)) for assignment in entry.cf_assignments))
                if may_be_a_method:
                    if node.self and function.is_attribute and isinstance(function.obj, ExprNodes.CloneNode) and (function.obj.arg is node.self):
                        function.obj = function.obj.arg
                    node = self.replace(node, ExprNodes.PyMethodCallNode.from_node(node, function=function, arg_tuple=node.arg_tuple, type=node.type))
        return node

    def visit_NumPyMethodCallNode(self, node):
        self.visitchildren(node)
        return node

    def visit_PyTypeTestNode(self, node):
        """Remove tests for alternatively allowed None values from
        type tests when we know that the argument cannot be None
        anyway.
        """
        self.visitchildren(node)
        if not node.notnone:
            if not node.arg.may_be_none():
                node.notnone = True
        return node

    def visit_NoneCheckNode(self, node):
        """Remove None checks from expressions that definitely do not
        carry a None value.
        """
        self.visitchildren(node)
        if not node.arg.may_be_none():
            return node.arg
        return node

    def visit_LoopNode(self, node):
        """Remember when we enter a loop as some expensive optimisations might still be worth it there.
        """
        old_val = self.in_loop
        self.in_loop = True
        self.visitchildren(node)
        self.in_loop = old_val
        return node

    def visit_IfStatNode(self, node):
        """Assign 'unlikely' branch hints to if-clauses that only raise exceptions.
        """
        self.visitchildren(node)
        last_non_unlikely_clause = None
        for i, if_clause in enumerate(node.if_clauses):
            self._set_ifclause_branch_hint(if_clause, if_clause.body)
            if not if_clause.branch_hint:
                last_non_unlikely_clause = if_clause
        if node.else_clause and last_non_unlikely_clause:
            self._set_ifclause_branch_hint(last_non_unlikely_clause, node.else_clause, inverse=True)
        return node

    def _set_ifclause_branch_hint(self, clause, statements_node, inverse=False):
        """Inject a branch hint if the if-clause unconditionally leads to a 'raise' statement.
        """
        if not statements_node.is_terminator:
            return
        non_branch_nodes = (Nodes.ExprStatNode, Nodes.AssignmentNode, Nodes.AssertStatNode, Nodes.DelStatNode, Nodes.GlobalNode, Nodes.NonlocalNode)
        statements = [statements_node]
        for next_node_pos, node in enumerate(statements, 1):
            if isinstance(node, Nodes.GILStatNode):
                statements.insert(next_node_pos, node.body)
                continue
            if isinstance(node, Nodes.StatListNode):
                statements[next_node_pos:next_node_pos] = node.stats
                continue
            if not isinstance(node, non_branch_nodes):
                if next_node_pos == len(statements) and isinstance(node, (Nodes.RaiseStatNode, Nodes.ReraiseStatNode)):
                    clause.branch_hint = 'likely' if inverse else 'unlikely'
                break