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
def _get_type_hint(autodoc_mock_imports: list[str], name: str, obj: Any) -> dict[str, Any]:
    _resolve_type_guarded_imports(autodoc_mock_imports, obj)
    try:
        result = get_type_hints(obj)
    except (AttributeError, TypeError, RecursionError) as exc:
        if isinstance(exc, TypeError) and _future_annotations_imported(obj) and ('unsupported operand type' in str(exc)):
            result = obj.__annotations__
        else:
            result = {}
    except NameError as exc:
        _LOGGER.warning('Cannot resolve forward reference in type annotations of "%s": %s', name, exc)
        result = obj.__annotations__
    return result