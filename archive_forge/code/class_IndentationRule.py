import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
class IndentationRule(Rule):
    code = 903

    def _get_message(self, message, node):
        message = super()._get_message(message, node)
        return 'IndentationError: ' + message