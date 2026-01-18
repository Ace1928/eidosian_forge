from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.common import UserError
from cmakelang.parse.util import COMMENT_TOKENS, WHITESPACE_TOKENS
from cmakelang.parse.common import NodeType, ParenBreaker, TreeNode
from cmakelang.parse.printer import tree_string
from cmakelang.parse.argument_nodes import StandardParser, StandardParser2
from cmakelang.parse.simple_nodes import CommentNode
class AtWordNode(TreeNode):

    def __init__(self):
        super(AtWordNode, self).__init__(NodeType.ATWORD)
        self.token = None

    @classmethod
    def parse(cls, ctx, tokens):
        node = cls()
        node.token = tokens.pop(0)
        node.children.append(node.token)
        return node