import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _any_fstring_error(version, node):
    if version < (3, 9) or node is None:
        return False
    if node.type == 'error_node':
        return any((child.type == 'fstring_start' for child in node.children))
    elif node.type == 'fstring':
        return True
    else:
        return node.search_ancestor('fstring')