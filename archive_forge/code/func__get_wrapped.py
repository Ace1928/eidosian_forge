from dataclasses import dataclass
from inspect import signature
from textwrap import dedent
import ast
import html
import inspect
import io as stdlib_io
import linecache
import os
import types
import warnings
from typing import (
import traitlets
from IPython.core import page
from IPython.lib.pretty import pretty
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import PyColorize, openpy
from IPython.utils.dir2 import safe_hasattr
from IPython.utils.path import compress_user
from IPython.utils.text import indent
from IPython.utils.wildcard import list_namespace, typestr2type
from IPython.utils.coloransi import TermColors
from IPython.utils.colorable import Colorable
from IPython.utils.decorators import undoc
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
def _get_wrapped(obj):
    """Get the original object if wrapped in one or more @decorators

    Some objects automatically construct similar objects on any unrecognised
    attribute access (e.g. unittest.mock.call). To protect against infinite loops,
    this will arbitrarily cut off after 100 levels of obj.__wrapped__
    attribute access. --TK, Jan 2016
    """
    orig_obj = obj
    i = 0
    while safe_hasattr(obj, '__wrapped__'):
        obj = obj.__wrapped__
        i += 1
        if i > 100:
            return orig_obj
    return obj