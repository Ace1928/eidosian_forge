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
class InlineDefNodeCalls(Visitor.NodeRefCleanupMixin, Visitor.EnvTransform):
    visit_Node = Visitor.VisitorTransform.recurse_to_children

    def get_constant_value_node(self, name_node):
        if name_node.cf_state is None:
            return None
        if name_node.cf_state.cf_is_null:
            return None
        entry = self.current_env().lookup(name_node.name)
        if not entry or (not entry.cf_assignments or len(entry.cf_assignments) != 1):
            return None
        return entry.cf_assignments[0].rhs

    def visit_SimpleCallNode(self, node):
        self.visitchildren(node)
        if not self.current_directives.get('optimize.inline_defnode_calls'):
            return node
        function_name = node.function
        if not function_name.is_name:
            return node
        function = self.get_constant_value_node(function_name)
        if not isinstance(function, ExprNodes.PyCFunctionNode):
            return node
        inlined = ExprNodes.InlinedDefNodeCallNode(node.pos, function_name=function_name, function=function, args=node.args, generator_arg_tag=node.generator_arg_tag)
        if inlined.can_be_inlined():
            return self.replace(node, inlined)
        return node