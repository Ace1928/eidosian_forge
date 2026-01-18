from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.common import InternalError
from cmakelang.parse.printer import dump_tree_tostr
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import CommentNode, OnOffNode
class KeywordGroupNode(TreeNode):
    """Argument subtree for a keyword and its arguments."""

    def __init__(self):
        super(KeywordGroupNode, self).__init__(NodeType.KWARGGROUP)
        self.keyword = None
        self.body = None

    @classmethod
    def parse(cls, ctx, tokens, word, subparser, breakstack):
        """
    Parse a standard `KWARG arg1 arg2 arg3...` style keyword argument list.
    """
        assert tokens[0].spelling.upper() == word.upper(), 'somehow dispatched wrong kwarg parse'
        tree = cls()
        keyword = KeywordNode.parse(ctx, tokens)
        tree.keyword = keyword
        tree.children.append(keyword)
        while tokens and tokens[0].type in WHITESPACE_TOKENS:
            tree.children.append(tokens.pop(0))
        ntokens = len(tokens)
        with ctx.pusharg(tree):
            subtree = subparser(ctx, tokens, breakstack)
        if len(tokens) < ntokens:
            tree.body = subtree
            tree.children.append(subtree)
        return tree