import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='sync_comp_for')
class _CompForRule(_CheckAssignmentRule):
    message = 'asynchronous comprehension outside of an asynchronous function'

    def is_issue(self, node):
        expr_list = node.children[1]
        if expr_list.type != 'expr_list':
            self._check_assignment(expr_list)
        return node.parent.children[0] == 'async' and (not self._normalizer.context.is_async_funcdef())