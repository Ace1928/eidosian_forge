import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(value='await')
class _AwaitOutsideAsync(SyntaxRule):
    message = "'await' outside async function"

    def is_issue(self, leaf):
        return not self._normalizer.context.is_async_funcdef()

    def get_error_node(self, node):
        return node.parent