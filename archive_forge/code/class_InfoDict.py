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
class InfoDict(TypedDict):
    type_name: Optional[str]
    base_class: Optional[str]
    string_form: Optional[str]
    namespace: Optional[str]
    length: Optional[str]
    file: Optional[str]
    definition: Optional[str]
    docstring: Optional[str]
    source: Optional[str]
    init_definition: Optional[str]
    class_docstring: Optional[str]
    init_docstring: Optional[str]
    call_def: Optional[str]
    call_docstring: Optional[str]
    subclasses: Optional[str]
    ismagic: bool
    isalias: bool
    isclass: bool
    found: bool
    name: str