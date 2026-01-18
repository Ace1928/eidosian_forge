from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import common
from cmakelang import lex
from cmakelang.parse.util import (
class TreeNode(object):
    """
  A node in the full-syntax-tree.
  """

    def __init__(self, node_type):
        self.node_type = node_type
        self.children = []
        self.parent = None

    def build_ancestry(self):
        """Recursively assign the .parent member within the subtree."""
        for child in self.children:
            if isinstance(child, TreeNode):
                child.parent = self
                child.build_ancestry()

    def get_location(self):
        """
    Return the (line, col) of the first token in the subtree rooted at this
    node.
    """
        if self.children:
            return self.children[0].get_location()
        return lex.SourceLocation((0, 0, 0))

    def count_newlines(self):
        newline_count = 0
        for child in self.children:
            newline_count += child.count_newlines()
        return newline_count

    def __repr__(self):
        return '{}: {}'.format(self.__class__.__name__, self.get_location())

    def get_tokens(self, out=None, kind=None):
        if out is None:
            out = []
        if kind is None:
            kind = 'all'
        match_group = {'semantic': is_semantic_token, 'syntactic': is_syntactic_token, 'whitespace': is_whitespace_token, 'comment': is_comment_token, 'all': lambda x: True}[kind]
        for child in self.children:
            if isinstance(child, lex.Token):
                if match_group(child):
                    out.append(child)
            elif isinstance(child, TreeNode):
                child.get_tokens(out, kind)
            else:
                raise RuntimeError('Unexpected child of type {}'.format(type(child)))
        return out

    def get_semantic_tokens(self, out=None):
        """
    Recursively reconstruct a stream of semantic tokens
    """
        return self.get_tokens(out, kind='semantic')