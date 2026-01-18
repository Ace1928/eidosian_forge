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
class ParenGroupNode(LayoutNode):
    """
  A parenthetical group. According to cmake syntax rules, this necessarily
  implies a boolean logical expression.
  """

    def __init__(self, pnode):
        super(ParenGroupNode, self).__init__(pnode)
        self._layout_passes = [(0, False), (1, False), (2, False), (3, False), (4, False), (5, True)]

    def has_terminal_comment(self):
        children = list(self.children)
        while children and children[0].node_type != NodeType.RPAREN:
            children.pop(0)
        if children:
            children.pop(0)
        return children and children[-1].pnode.children[0].type == TokenType.COMMENT

    def _reflow(self, stack_context, cursor, passno):
        config = stack_context.config
        children = list(self.children)
        self._colextent = cursor.y
        assert len(children) in (3, 4)
        prev = None
        child = None
        column_cursor = cursor.clone()
        while children:
            prev = child
            child = children.pop(0)
            if prev is None:
                pass
            elif prev.node_type == NodeType.LPAREN:
                pass
            elif is_line_comment(prev) or prev.has_terminal_comment() or self._wrap:
                cursor[1] = column_cursor[1]
                cursor[0] += 1
            elif child.node_type == NodeType.RPAREN:
                pass
            else:
                cursor[1] += 1
            if self.statement_terminal and (not children):
                child.statement_terminal = True
            cursor = child.reflow(stack_context, cursor, passno)
            if not self._wrap:
                needs_wrap = False
                if child.statement_terminal:
                    if cursor[1] + 1 > config.format.linewidth:
                        needs_wrap = True
                if child.colextent > config.format.linewidth:
                    needs_wrap = True
                if not child.reflow_valid:
                    needs_wrap = True
                if needs_wrap:
                    column_cursor[0] += 1
                    cursor = Cursor(*column_cursor)
                    cursor = child.reflow(stack_context, cursor, passno)
            self._reflow_valid &= child.reflow_valid
            self._colextent = max(self._colextent, child.colextent)
            column_cursor[0] = cursor[0]
        return cursor

    def write(self, config, ctx):
        if not ctx.is_active():
            return
        super(ParenGroupNode, self).write(config, ctx)