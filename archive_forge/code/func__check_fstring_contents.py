import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _check_fstring_contents(self, children, depth=0):
    for fstring_content in children:
        if fstring_content.type == 'fstring_expr':
            self._check_fstring_expr(fstring_content, depth)