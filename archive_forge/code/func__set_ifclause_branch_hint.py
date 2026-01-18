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