import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _add_syntax_error(self, node, message):
    self.add_issue(node, 901, 'SyntaxError: ' + message)