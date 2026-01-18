import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _get_comprehension_type(atom):
    first, second = atom.children[:2]
    if second.type == 'testlist_comp' and second.children[1].type in _COMP_FOR_TYPES:
        if first == '[':
            return 'list comprehension'
        else:
            return 'generator expression'
    elif second.type == 'dictorsetmaker' and second.children[-1].type in _COMP_FOR_TYPES:
        if second.children[1] == ':':
            return 'dict comprehension'
        else:
            return 'set comprehension'
    return None