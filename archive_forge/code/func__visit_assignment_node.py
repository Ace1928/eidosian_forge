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
def _visit_assignment_node(self, node, expr_list):
    """Flatten parallel assignments into separate single
        assignments or cascaded assignments.
        """
    if sum([1 for expr in expr_list if expr.is_sequence_constructor or expr.is_string_literal]) < 2:
        return node
    expr_list_list = []
    flatten_parallel_assignments(expr_list, expr_list_list)
    temp_refs = []
    eliminate_rhs_duplicates(expr_list_list, temp_refs)
    nodes = []
    for expr_list in expr_list_list:
        lhs_list = expr_list[:-1]
        rhs = expr_list[-1]
        if len(lhs_list) == 1:
            node = Nodes.SingleAssignmentNode(rhs.pos, lhs=lhs_list[0], rhs=rhs)
        else:
            node = Nodes.CascadedAssignmentNode(rhs.pos, lhs_list=lhs_list, rhs=rhs)
        nodes.append(node)
    if len(nodes) == 1:
        assign_node = nodes[0]
    else:
        assign_node = Nodes.ParallelAssignmentNode(nodes[0].pos, stats=nodes)
    if temp_refs:
        duplicates_and_temps = [(temp.expression, temp) for temp in temp_refs]
        sort_common_subsequences(duplicates_and_temps)
        for _, temp_ref in duplicates_and_temps[::-1]:
            assign_node = LetNode(temp_ref, assign_node)
    return assign_node