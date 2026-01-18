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
class FlattenInListTransform(Visitor.VisitorTransform, SkipDeclarations):
    """
    This transformation flattens "x in [val1, ..., valn]" into a sequential list
    of comparisons.
    """

    def visit_PrimaryCmpNode(self, node):
        self.visitchildren(node)
        if node.cascade is not None:
            return node
        elif node.operator == 'in':
            conjunction = 'or'
            eq_or_neq = '=='
        elif node.operator == 'not_in':
            conjunction = 'and'
            eq_or_neq = '!='
        else:
            return node
        if not isinstance(node.operand2, (ExprNodes.TupleNode, ExprNodes.ListNode, ExprNodes.SetNode)):
            return node
        args = node.operand2.args
        if len(args) == 0:
            return node
        if any([arg.is_starred for arg in args]):
            return node
        lhs = UtilNodes.ResultRefNode(node.operand1)
        conds = []
        temps = []
        for arg in args:
            try:
                is_simple_arg = arg.is_simple()
            except Exception:
                is_simple_arg = False
            if not is_simple_arg:
                arg = UtilNodes.LetRefNode(arg)
                temps.append(arg)
            cond = ExprNodes.PrimaryCmpNode(pos=node.pos, operand1=lhs, operator=eq_or_neq, operand2=arg, cascade=None)
            conds.append(ExprNodes.TypecastNode(pos=node.pos, operand=cond, type=PyrexTypes.c_bint_type))

        def concat(left, right):
            return ExprNodes.BoolBinopNode(pos=node.pos, operator=conjunction, operand1=left, operand2=right)
        condition = reduce(concat, conds)
        new_node = UtilNodes.EvalWithTempExprNode(lhs, condition)
        for temp in temps[::-1]:
            new_node = UtilNodes.EvalWithTempExprNode(temp, new_node)
        return new_node
    visit_Node = Visitor.VisitorTransform.recurse_to_children