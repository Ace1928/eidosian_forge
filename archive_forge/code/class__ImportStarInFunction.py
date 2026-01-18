import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='import_from')
class _ImportStarInFunction(SyntaxRule):
    message = 'import * only allowed at module level'

    def is_issue(self, node):
        return node.is_star_import() and self._normalizer.context.parent_context is not None