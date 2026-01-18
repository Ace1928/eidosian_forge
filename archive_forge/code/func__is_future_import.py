import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _is_future_import(import_from):
    from_names = import_from.get_from_names()
    return [n.value for n in from_names] == ['__future__']