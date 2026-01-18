from __future__ import annotations
import inspect
import re
import sys
import textwrap
import types
from ast import FunctionDef, Module, stmt
from dataclasses import dataclass
from typing import Any, AnyStr, Callable, ForwardRef, NewType, TypeVar, get_type_hints
from docutils.frontend import OptionParser
from docutils.nodes import Node
from docutils.parsers.rst import Parser as RstParser
from docutils.utils import new_document
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import Options
from sphinx.ext.autodoc.mock import mock
from sphinx.util import logging
from sphinx.util.inspect import signature as sphinx_signature
from sphinx.util.inspect import stringify_signature
from .patches import install_patches
from .version import __version__
def _get_sphinx_line_keyword_and_argument(line: str) -> tuple[str, str | None] | None:
    """
    Extract a keyword, and its optional argument out of a sphinx field option line.

    For example
    >>> _get_sphinx_line_keyword_and_argument(":param parameter:")
    ("param", "parameter")
    >>> _get_sphinx_line_keyword_and_argument(":return:")
    ("return", None)
    >>> _get_sphinx_line_keyword_and_argument("some invalid line")
    None
    """
    param_line_without_description = line.split(':', maxsplit=2)
    if len(param_line_without_description) != 3:
        return None
    split_directive_and_name = param_line_without_description[1].split(maxsplit=1)
    if len(split_directive_and_name) != 2:
        if not len(split_directive_and_name):
            return None
        return (split_directive_and_name[0], None)
    return tuple(split_directive_and_name)