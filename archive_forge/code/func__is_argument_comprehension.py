import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _is_argument_comprehension(argument):
    return argument.children[1].type in _COMP_FOR_TYPES