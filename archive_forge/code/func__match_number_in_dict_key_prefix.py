from __future__ import annotations
import builtins as builtin_mod
import enum
import glob
import inspect
import itertools
import keyword
import os
import re
import string
import sys
import tokenize
import time
import unicodedata
import uuid
import warnings
from ast import literal_eval
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from types import SimpleNamespace
from typing import (
from IPython.core.guarded_eval import guarded_eval, EvaluationContext
from IPython.core.error import TryNext
from IPython.core.inputtransformer2 import ESC_MAGIC
from IPython.core.latex_symbols import latex_symbols, reverse_latex_symbol
from IPython.core.oinspect import InspectColors
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import generics
from IPython.utils.decorators import sphinx_options
from IPython.utils.dir2 import dir2, get_real_method
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.path import ensure_dir_exists
from IPython.utils.process import arg_split
from traitlets import (
from traitlets.config.configurable import Configurable
import __main__
def _match_number_in_dict_key_prefix(prefix: str) -> Union[str, None]:
    """Match any valid Python numeric literal in a prefix of dictionary keys.

    References:
    - https://docs.python.org/3/reference/lexical_analysis.html#numeric-literals
    - https://docs.python.org/3/library/tokenize.html
    """
    if prefix[-1].isspace():
        return None
    tokens = _parse_tokens(prefix)
    rev_tokens = reversed(tokens)
    skip_over = {tokenize.ENDMARKER, tokenize.NEWLINE}
    number = None
    for token in rev_tokens:
        if token.type in skip_over:
            continue
        if number is None:
            if token.type == tokenize.NUMBER:
                number = token.string
                continue
            else:
                return None
        if token.type == tokenize.OP:
            if token.string == ',':
                break
            if token.string in {'+', '-'}:
                number = token.string + number
        else:
            return None
    return number