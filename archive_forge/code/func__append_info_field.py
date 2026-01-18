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
def _append_info_field(self, bundle: UnformattedBundle, title: str, key: str, info, omit_sections: List[str], formatter):
    """Append an info value to the unformatted mimebundle being constructed by _make_info_unformatted"""
    if title in omit_sections or key in omit_sections:
        return
    field = info[key]
    if field is not None:
        formatted_field = self._mime_format(field, formatter)
        bundle['text/plain'].append((title, formatted_field['text/plain']))
        bundle['text/html'].append((title, formatted_field['text/html']))