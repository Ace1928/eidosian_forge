from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import io
import logging
import re
import sys
from cmakelang import lex
from cmakelang import markup
from cmakelang.common import UserError
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import PositionalGroupNode
from cmakelang.parse.common import FlowType, NodeType, TreeNode
from cmakelang.parse.util import comment_is_tag
from cmakelang.parse import simple_nodes
class ParenNode(LayoutNode):
    """Holds parenthesis '(' or ')' for statements or boolean groups."""

    @property
    def name(self):
        return self.node_type.name

    def _reflow(self, stack_context, cursor, passno):
        """There is only one possible layout for this node."""
        self._colextent = cursor[1] + 1
        return cursor + (0, 1)

    def write(self, config, ctx):
        if self.node_type == NodeType.LPAREN:
            ctx.outfile.write_at(self.position, '(')
        elif self.node_type == NodeType.RPAREN:
            ctx.outfile.write_at(self.position, ')')
        else:
            raise ValueError('Unrecognized paren type')