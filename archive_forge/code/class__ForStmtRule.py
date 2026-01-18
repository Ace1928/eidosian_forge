import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='for_stmt')
class _ForStmtRule(_CheckAssignmentRule):

    def is_issue(self, for_stmt):
        expr_list = for_stmt.children[1]
        if expr_list.type != 'expr_list':
            self._check_assignment(expr_list)