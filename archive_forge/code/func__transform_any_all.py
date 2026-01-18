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
def _transform_any_all(self, node, pos_args, is_any):
    if len(pos_args) != 1:
        return node
    if not isinstance(pos_args[0], ExprNodes.GeneratorExpressionNode):
        return node
    gen_expr_node = pos_args[0]
    generator_body = gen_expr_node.def_node.gbody
    loop_node = generator_body.body
    yield_expression, yield_stat_node = _find_single_yield_expression(loop_node)
    if yield_expression is None:
        return node
    if is_any:
        condition = yield_expression
    else:
        condition = ExprNodes.NotNode(yield_expression.pos, operand=yield_expression)
    test_node = Nodes.IfStatNode(yield_expression.pos, else_clause=None, if_clauses=[Nodes.IfClauseNode(yield_expression.pos, condition=condition, body=Nodes.ReturnStatNode(node.pos, value=ExprNodes.BoolNode(yield_expression.pos, value=is_any, constant_result=is_any)))])
    loop_node.else_clause = Nodes.ReturnStatNode(node.pos, value=ExprNodes.BoolNode(yield_expression.pos, value=not is_any, constant_result=not is_any))
    Visitor.recursively_replace_node(gen_expr_node, yield_stat_node, test_node)
    return ExprNodes.InlinedGeneratorExpressionNode(gen_expr_node.pos, gen=gen_expr_node, orig_func='any' if is_any else 'all')