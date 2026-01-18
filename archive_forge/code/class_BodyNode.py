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
class BodyNode(LayoutNode):
    """
  Top-level node for a given "scope" depth. This node is the root of a document,
  or the root of any nested statement scopes.
  """

    def _reflow(self, stack_context, cursor, passno):
        """
    Compute the size of a body block
    """
        column_cursor = cursor.clone()
        self._colextent = 0
        for child in self.children:
            cursor = child.reflow(stack_context, column_cursor, passno)
            self._reflow_valid &= child.reflow_valid
            self._colextent = max(self._colextent, child.colextent)
            column_cursor[0] = cursor[0] + 1
        return cursor