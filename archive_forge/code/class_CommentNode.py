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
class CommentNode(LayoutNode):
    """
  A line comment or bracket comment. If parented by a group node then this
  comment acts as an argument. If parented by a scalar node, then this comment
  acts like an argument comment.
  """

    def __init__(self, pnode):
        super(CommentNode, self).__init__(pnode)
        self._lines = []

    def is_tag(self):
        return comment_is_tag(self.pnode.children[0])

    def _reflow(self, stack_context, cursor, passno):
        """
    Compute the size of a comment block
    """
        config = stack_context.config
        if len(self.pnode.children) == 1 and isinstance(self.pnode.children[0], lex.Token) and (self.pnode.children[0].type == TokenType.BRACKET_COMMENT):
            content = normalize_line_endings(self.pnode.children[0].spelling)
            lines = content.split('\n')
            line = lines.pop(0)
            cursor[1] += len(line)
            self._colextent = cursor[1]
            while lines:
                cursor[0] += 1
                cursor[1] = 0
                line = lines.pop(0)
                cursor[1] += len(line)
                self._colextent = max(self._colextent, cursor[1])
            return cursor
        allocation = config.format.linewidth - cursor[1]
        with stack_context.push_node(self):
            self._lines = lines = list(format_comment_lines(self.pnode, stack_context, allocation))
        self._colextent = cursor[1] + max((len(line) for line in lines))
        increment = (len(lines) - 1, len(lines[-1]))
        return cursor + increment

    def write(self, config, ctx):
        if not ctx.is_active():
            return
        if len(self.pnode.children) == 1 and isinstance(self.pnode.children[0], lex.Token) and (self.pnode.children[0].type == TokenType.BRACKET_COMMENT):
            content = normalize_line_endings(self.pnode.children[0].spelling)
            ctx.outfile.write_at(self.position, content)
        else:
            for idx, line in enumerate(self._lines):
                ctx.outfile.write_at(self.position + (idx, 0), line)