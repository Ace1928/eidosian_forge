import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='argument')
class _ArgumentRule(SyntaxRule):

    def is_issue(self, node):
        first = node.children[0]
        if self._normalizer.version < (3, 8):
            first = _remove_parens(first)
        if node.children[1] == '=' and first.type != 'name':
            if first.type == 'lambdef':
                if self._normalizer.version < (3, 8):
                    message = 'lambda cannot contain assignment'
                else:
                    message = 'expression cannot contain assignment, perhaps you meant "=="?'
            elif self._normalizer.version < (3, 8):
                message = "keyword can't be an expression"
            else:
                message = 'expression cannot contain assignment, perhaps you meant "=="?'
            self.add_issue(first, message=message)
        if _is_argument_comprehension(node) and node.parent.type == 'classdef':
            self.add_issue(node, message='invalid syntax')