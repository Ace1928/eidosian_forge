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
def _transform_indexable_iteration(self, node, slice_node, is_mutable, reversed=False):
    """In principle can handle any iterable that Cython has a len() for and knows how to index"""
    unpack_temp_node = UtilNodes.LetRefNode(slice_node.as_none_safe_node("'NoneType' is not iterable"), may_hold_none=False, is_temp=True)
    start_node = ExprNodes.IntNode(node.pos, value='0', constant_result=0, type=PyrexTypes.c_py_ssize_t_type)

    def make_length_call():
        builtin_len = ExprNodes.NameNode(node.pos, name='len', entry=Builtin.builtin_scope.lookup('len'))
        return ExprNodes.SimpleCallNode(node.pos, function=builtin_len, args=[unpack_temp_node])
    length_temp = UtilNodes.LetRefNode(make_length_call(), type=PyrexTypes.c_py_ssize_t_type, is_temp=True)
    end_node = length_temp
    if reversed:
        relation1, relation2 = ('>', '>=')
        start_node, end_node = (end_node, start_node)
    else:
        relation1, relation2 = ('<=', '<')
    counter_ref = UtilNodes.LetRefNode(pos=node.pos, type=PyrexTypes.c_py_ssize_t_type)
    target_value = ExprNodes.IndexNode(slice_node.pos, base=unpack_temp_node, index=counter_ref)
    target_assign = Nodes.SingleAssignmentNode(pos=node.target.pos, lhs=node.target, rhs=target_value)
    env = self.current_env()
    new_directives = Options.copy_inherited_directives(env.directives, boundscheck=False, wraparound=False)
    target_assign = Nodes.CompilerDirectivesNode(target_assign.pos, directives=new_directives, body=target_assign)
    body = Nodes.StatListNode(node.pos, stats=[target_assign])
    if is_mutable:
        loop_length_reassign = Nodes.SingleAssignmentNode(node.pos, lhs=length_temp, rhs=make_length_call())
        body.stats.append(loop_length_reassign)
    loop_node = Nodes.ForFromStatNode(node.pos, bound1=start_node, relation1=relation1, target=counter_ref, relation2=relation2, bound2=end_node, step=None, body=body, else_clause=node.else_clause, from_range=True)
    ret = UtilNodes.LetNode(unpack_temp_node, UtilNodes.LetNode(length_temp, Nodes.ExprStatNode(node.pos, expr=UtilNodes.TempResultFromStatNode(counter_ref, loop_node)))).analyse_expressions(env)
    body.stats.insert(1, node.body)
    return ret