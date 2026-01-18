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
def _default_string_list_lexer(value):
    """Simple string tokenizer for lists of words.

    This default lexer splits strings on whitespace and/or commas while
    honoring use of single and double quotes.  Separators (whitespace or
    commas) are not returned.  Consecutive delimiters are ignored (and
    do not yield empty strings).

    """
    _lex = _default_string_list_lexer._lex
    if _lex is None:
        _default_string_list_lexer._lex = _lex = _build_lexer()
    _lex.input(value)
    while True:
        tok = _lex.token()
        if not tok:
            break
        yield tok.value