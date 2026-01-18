import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _get_message(self, message, node):
    message = super()._get_message(message, node)
    if 'f-string' not in message and _any_fstring_error(self._normalizer.version, node):
        message = 'f-string: ' + message
    return 'SyntaxError: ' + message