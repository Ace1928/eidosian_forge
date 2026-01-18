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
def _default_arguments_from_docstring(self, doc):
    """Parse the first line of docstring for call signature.

        Docstring should be of the form 'min(iterable[, key=func])
'.
        It can also parse cython docstring of the form
        'Minuit.migrad(self, int ncall=10000, resume=True, int nsplit=1)'.
        """
    if doc is None:
        return []
    line = doc.lstrip().splitlines()[0]
    sig = self.docstring_sig_re.search(line)
    if sig is None:
        return []
    sig = sig.groups()[0].split(',')
    ret = []
    for s in sig:
        ret += self.docstring_kwd_re.findall(s)
    return ret