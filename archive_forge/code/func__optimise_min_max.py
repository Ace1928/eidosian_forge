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
def _optimise_min_max(self, node, args, operator):
    """Replace min(a,b,...) and max(a,b,...) by explicit comparison code.
        """
    if len(args) <= 1:
        if len(args) == 1 and args[0].is_sequence_constructor:
            args = args[0].args
        if len(args) <= 1:
            return node
    cascaded_nodes = list(map(UtilNodes.ResultRefNode, args[1:]))
    last_result = args[0]
    for arg_node in cascaded_nodes:
        result_ref = UtilNodes.ResultRefNode(last_result)
        last_result = ExprNodes.CondExprNode(arg_node.pos, true_val=arg_node, false_val=result_ref, test=ExprNodes.PrimaryCmpNode(arg_node.pos, operand1=arg_node, operator=operator, operand2=result_ref))
        last_result = UtilNodes.EvalWithTempExprNode(result_ref, last_result)
    for ref_node in cascaded_nodes[::-1]:
        last_result = UtilNodes.EvalWithTempExprNode(ref_node, last_result)
    return last_result