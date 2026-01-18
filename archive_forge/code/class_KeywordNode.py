from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.common import InternalError
from cmakelang.parse.printer import dump_tree_tostr
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import CommentNode, OnOffNode
class KeywordNode(TreeNode):
    """Node that stores a single keyword token and possibly it's associated
     comments."""

    def __init__(self):
        super(KeywordNode, self).__init__(NodeType.KEYWORD)
        self.token = None

    @classmethod
    def parse(cls, _ctx, tokens):
        node = cls()
        node.token = tokens.pop(0)
        node.children.append(node.token)
        return node