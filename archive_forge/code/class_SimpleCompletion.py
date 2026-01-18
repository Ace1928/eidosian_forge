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
class SimpleCompletion:
    """Completion item to be included in the dictionary returned by new-style Matcher (API v2).

    .. warning::

        Provisional

        This class is used to describe the currently supported attributes of
        simple completion items, and any additional implementation details
        should not be relied on. Additional attributes may be included in
        future versions, and meaning of text disambiguated from the current
        dual meaning of "text to insert" and "text to used as a label".
    """
    __slots__ = ['text', 'type']

    def __init__(self, text: str, *, type: Optional[str]=None):
        self.text = text
        self.type = type

    def __repr__(self):
        return f'<SimpleCompletion text={self.text!r} type={self.type!r}>'