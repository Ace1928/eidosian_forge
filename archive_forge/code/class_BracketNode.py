import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
class BracketNode(IndentationNode):

    def __init__(self, config, leaf, parent, in_suite_introducer=False):
        self.leaf = leaf
        previous_leaf = leaf
        n = parent
        if n.type == IndentationTypes.IMPLICIT:
            n = n.parent
        while True:
            if hasattr(n, 'leaf') and previous_leaf.line != n.leaf.line:
                break
            previous_leaf = previous_leaf.get_previous_leaf()
            if not isinstance(n, BracketNode) or previous_leaf != n.leaf:
                break
            n = n.parent
        parent_indentation = n.indentation
        next_leaf = leaf.get_next_leaf()
        if '\n' in next_leaf.prefix or '\r' in next_leaf.prefix:
            self.bracket_indentation = parent_indentation + config.closing_bracket_hanging_indentation
            self.indentation = parent_indentation + config.indentation
            self.type = IndentationTypes.HANGING_BRACKET
        else:
            expected_end_indent = leaf.end_pos[1]
            if '\t' in config.indentation:
                self.indentation = None
            else:
                self.indentation = ' ' * expected_end_indent
            self.bracket_indentation = self.indentation
            self.type = IndentationTypes.VERTICAL_BRACKET
        if in_suite_introducer and parent.type == IndentationTypes.SUITE and (self.indentation == parent_indentation + config.indentation):
            self.indentation += config.indentation
            self.bracket_indentation = self.indentation
        self.parent = parent