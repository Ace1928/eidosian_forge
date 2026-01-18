from builtins import open as _builtin_open
from codecs import lookup, BOM_UTF8
import collections
import functools
from io import TextIOWrapper
import itertools as _itertools
import re
import sys
from token import *
from token import EXACT_TOKEN_TYPES
import token
def _generate_tokens_from_c_tokenizer(source):
    """Tokenize a source reading Python code as unicode strings using the internal C tokenizer"""
    import _tokenize as c_tokenizer
    for info in c_tokenizer.TokenizerIter(source):
        tok, type, lineno, end_lineno, col_off, end_col_off, line = info
        yield TokenInfo(type, tok, (lineno, col_off), (end_lineno, end_col_off), line)