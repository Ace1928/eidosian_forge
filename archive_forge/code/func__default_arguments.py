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
def _default_arguments(self, obj):
    """Return the list of default arguments of obj if it is callable,
        or empty list otherwise."""
    call_obj = obj
    ret = []
    if inspect.isbuiltin(obj):
        pass
    elif not (inspect.isfunction(obj) or inspect.ismethod(obj)):
        if inspect.isclass(obj):
            ret += self._default_arguments_from_docstring(getattr(obj, '__doc__', ''))
            call_obj = getattr(obj, '__init__', None) or getattr(obj, '__new__', None)
        elif hasattr(obj, '__call__'):
            call_obj = obj.__call__
    ret += self._default_arguments_from_docstring(getattr(call_obj, '__doc__', ''))
    _keeps = (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    try:
        sig = inspect.signature(obj)
        ret.extend((k for k, v in sig.parameters.items() if v.kind in _keeps))
    except ValueError:
        pass
    return list(set(ret))