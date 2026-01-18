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
def _resolve_type_guarded_imports(autodoc_mock_imports: list[str], obj: Any) -> None:
    if hasattr(obj, '__module__') and obj.__module__ not in _TYPE_GUARD_IMPORTS_RESOLVED or (hasattr(obj, '__globals__') and id(obj.__globals__) not in _TYPE_GUARD_IMPORTS_RESOLVED_GLOBALS_ID):
        _TYPE_GUARD_IMPORTS_RESOLVED.add(obj.__module__)
        if obj.__module__ not in sys.builtin_module_names:
            if hasattr(obj, '__globals__'):
                _TYPE_GUARD_IMPORTS_RESOLVED_GLOBALS_ID.add(id(obj.__globals__))
            module = inspect.getmodule(obj)
            if module:
                try:
                    module_code = inspect.getsource(module)
                except (TypeError, OSError):
                    ...
                else:
                    for _, part in _TYPE_GUARD_IMPORT_RE.findall(module_code):
                        guarded_code = textwrap.dedent(part)
                        try:
                            with mock(autodoc_mock_imports):
                                exec(guarded_code, obj.__globals__)
                        except Exception as exc:
                            _LOGGER.warning(f'Failed guarded type import with {exc!r}')