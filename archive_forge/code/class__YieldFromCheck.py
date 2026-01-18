import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(value='from')
class _YieldFromCheck(SyntaxRule):
    message = "'yield from' inside async function"

    def get_node(self, leaf):
        return leaf.parent.parent

    def is_issue(self, leaf):
        return leaf.parent.type == 'yield_arg' and self._normalizer.context.is_async_funcdef()