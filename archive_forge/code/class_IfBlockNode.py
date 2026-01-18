from __future__ import print_function
from __future__ import unicode_literals
import collections
import logging
from cmakelang.common import InternalError
from cmakelang import lex
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import (
from cmakelang.parse.statement_node import (
class IfBlockNode(FlowControlNode):
    """IfBlocks are different than other `FlowControlNode` s in that they can
     contain multiple bodies.
  """

    @classmethod
    def consume(cls, ctx, tokens):
        """
    Consume tokens and return a flow control tree. ``IF`` statements are special
    because they have interior ``ELSIF`` and ``ELSE`` blocks, while all other
    flow control have a single body.
    """
        tree = cls()
        breakset = ('ELSE', 'ELSEIF', 'ENDIF')
        prev_open_stmt = None
        prev_body = None
        while tokens and tokens[0].spelling.upper() != 'ENDIF':
            open_stmt = StatementNode.consume(ctx, tokens)
            body = BodyNode.consume(ctx, tokens, breakset)
            tree.children.append(open_stmt)
            tree.children.append(body)
            if prev_open_stmt is not None:
                tree.blocks.append(FlowControlBlock(prev_open_stmt, prev_body, open_stmt))
            prev_open_stmt = open_stmt
            prev_body = body
        if tokens:
            close_stmt = StatementNode.consume(ctx, tokens)
            tree.children.append(close_stmt)
            tree.blocks.append(FlowControlBlock(prev_open_stmt, prev_body, close_stmt))
        return tree