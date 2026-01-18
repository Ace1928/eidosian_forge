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
def _handle_simple_method_unicode_join(self, node, function, args, is_unbound_method):
    """
        unicode.join() builds a list first => see if we can do this more efficiently
        """
    if len(args) != 2:
        self._error_wrong_arg_count('unicode.join', node, args, '2')
        return node
    if isinstance(args[1], ExprNodes.GeneratorExpressionNode):
        gen_expr_node = args[1]
        loop_node = gen_expr_node.loop
        yield_statements = _find_yield_statements(loop_node)
        if yield_statements:
            inlined_genexpr = ExprNodes.InlinedGeneratorExpressionNode(node.pos, gen_expr_node, orig_func='list', comprehension_type=Builtin.list_type)
            for yield_expression, yield_stat_node in yield_statements:
                append_node = ExprNodes.ComprehensionAppendNode(yield_expression.pos, expr=yield_expression, target=inlined_genexpr.target)
                Visitor.recursively_replace_node(gen_expr_node, yield_stat_node, append_node)
            args[1] = inlined_genexpr
    return self._substitute_method_call(node, function, 'PyUnicode_Join', self.PyUnicode_Join_func_type, 'join', is_unbound_method, args)