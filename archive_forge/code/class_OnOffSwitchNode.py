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
class OnOffSwitchNode(LayoutNode):
    """
  Holds a special-case line comment token such as ``# cmake-format: off`` or
  ``# cmake-format: on``
  """

    @property
    def name(self):
        return self.pnode.children[0].type.name

    def has_terminal_comment(self):
        return True

    def _reflow(self, stack_context, cursor, passno):
        """
    There is only one possible flow as this is a single-token
    """
        assert self.pnode.children
        token = self.pnode.children[0]
        assert isinstance(token, lex.Token)
        if token.type == TokenType.FORMAT_ON:
            spelling = '# cmake-format: on'
        elif token.type == TokenType.FORMAT_OFF:
            spelling = '# cmake-format: off'
        cursor = cursor + (0, len(spelling))
        self._colextent = cursor[1]
        return cursor

    def write(self, config, ctx):
        assert self.pnode.children
        token = self.pnode.children[0]
        assert isinstance(token, lex.Token)
        if token.type == TokenType.FORMAT_ON:
            spelling = '# cmake-format: on'
            if ctx.offswitch_location is None:
                logging.warning("'#cmake-format: on' with no corresponding 'off' at %d:%d", token.begin.line, token.begin.col)
            else:
                ctx.infile.seek(ctx.offswitch_location.offset, 0)
                copy_size = token.begin.offset - ctx.offswitch_location.offset
                copy_bytes = ctx.infile.read(copy_size)
                copy_text = copy_bytes.decode('utf-8')
                ctx.outfile.write(copy_text)
                ctx.offswitch_location = None
                ctx.outfile.forge_cursor(self.position)
        elif token.type == TokenType.FORMAT_OFF:
            spelling = '# cmake-format: off'
            ctx.offswitch_location = token.end
        ctx.outfile.write_at(self.position, spelling)