import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _check_format_spec(self, format_spec, depth):
    self._check_fstring_contents(format_spec.children[1:], depth)