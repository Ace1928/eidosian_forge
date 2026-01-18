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
class ReplaceFusedTypeChecks(VisitorTransform):
    """
    This is not a transform in the pipeline. It is invoked on the specific
    versions of a cdef function with fused argument types. It filters out any
    type branches that don't match. e.g.

        if fused_t is mytype:
            ...
        elif fused_t in other_fused_type:
            ...
    """

    def __init__(self, local_scope):
        super(ReplaceFusedTypeChecks, self).__init__()
        self.local_scope = local_scope
        from .Optimize import ConstantFolding
        self.transform = ConstantFolding(reevaluate=True)

    def visit_IfStatNode(self, node):
        """
        Filters out any if clauses with false compile time type check
        expression.
        """
        self.visitchildren(node)
        return self.transform(node)

    def visit_GILStatNode(self, node):
        """
        Fold constant condition of GILStatNode.
        """
        self.visitchildren(node)
        return self.transform(node)

    def visit_PrimaryCmpNode(self, node):
        with Errors.local_errors(ignore=True):
            type1 = node.operand1.analyse_as_type(self.local_scope)
            type2 = node.operand2.analyse_as_type(self.local_scope)
        if type1 and type2:
            false_node = ExprNodes.BoolNode(node.pos, value=False)
            true_node = ExprNodes.BoolNode(node.pos, value=True)
            type1 = self.specialize_type(type1, node.operand1.pos)
            op = node.operator
            if op in ('is', 'is_not', '==', '!='):
                type2 = self.specialize_type(type2, node.operand2.pos)
                is_same = type1.same_as(type2)
                eq = op in ('is', '==')
                if is_same and eq or (not is_same and (not eq)):
                    return true_node
            elif op in ('in', 'not_in'):
                if isinstance(type2, PyrexTypes.CTypedefType):
                    type2 = type2.typedef_base_type
                if type1.is_fused:
                    error(node.operand1.pos, 'Type is fused')
                elif not type2.is_fused:
                    error(node.operand2.pos, "Can only use 'in' or 'not in' on a fused type")
                else:
                    types = PyrexTypes.get_specialized_types(type2)
                    for specialized_type in types:
                        if type1.same_as(specialized_type):
                            if op == 'in':
                                return true_node
                            else:
                                return false_node
                    if op == 'not_in':
                        return true_node
            return false_node
        return node

    def specialize_type(self, type, pos):
        try:
            return type.specialize(self.local_scope.fused_to_specific)
        except KeyError:
            error(pos, 'Type is not specific')
            return type

    def visit_Node(self, node):
        self.visitchildren(node)
        return node