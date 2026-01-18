import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='del_stmt')
class _DelStmtRule(_CheckAssignmentRule):

    def is_issue(self, del_stmt):
        child = del_stmt.children[1]
        if child.type != 'expr_list':
            self._check_assignment(child, is_deletion=True)