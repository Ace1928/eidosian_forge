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
def _mime_format(self, text: str, formatter=None) -> dict:
    """Return a mime bundle representation of the input text.

        - if `formatter` is None, the returned mime bundle has
           a ``text/plain`` field, with the input text.
           a ``text/html`` field with a ``<pre>`` tag containing the input text.

        - if ``formatter`` is not None, it must be a callable transforming the
          input text into a mime bundle. Default values for ``text/plain`` and
          ``text/html`` representations are the ones described above.

        Note:

        Formatters returning strings are supported but this behavior is deprecated.

        """
    defaults = {'text/plain': text, 'text/html': f'<pre>{html.escape(text)}</pre>'}
    if formatter is None:
        return defaults
    else:
        formatted = formatter(text)
        if not isinstance(formatted, dict):
            return {'text/plain': formatted, 'text/html': f'<pre>{formatted}</pre>'}
        else:
            return dict(defaults, **formatted)