import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='with_item')
class _WithItemRule(_CheckAssignmentRule):

    def is_issue(self, with_item):
        self._check_assignment(with_item.children[2])