import argparse
import builtins
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types
from operator import attrgetter
from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
from pyomo.common.fileutils import import_file
from pyomo.common.formatting import wrap_reStructuredText
from pyomo.common.modeling import NOTSET
def _default_string_dict_lexer(value):
    """Simple string tokenizer for dict data.

    This default lexer splits strings on whitespace and/or commas while
    honoring use of single and double quotes.  ':' and '=' are
    recognized as special tokens.  Separators (whitespace or commas) are
    not returned.  Consecutive delimiters are ignored (and do not yield
    empty strings).

    """
    _lex = _default_string_dict_lexer._lex
    if _lex is None:
        _default_string_dict_lexer._lex = _lex = _build_lexer(':=')
    _lex.input(value)
    while True:
        key = _lex.token()
        if not key:
            break
        sep = _lex.token()
        if not sep:
            raise ValueError("Expected ':' or '=' but encountered end of string")
        if sep.type not in ':=':
            raise ValueError(f"Expected ':' or '=' but found '{sep.value}' at Line {sep.lineno} Column {sep.lexpos + 1}")
        val = _lex.token()
        if not val:
            raise ValueError(f"Expected value following '{sep.type}' but encountered end of string")
        yield (key.value, val.value)