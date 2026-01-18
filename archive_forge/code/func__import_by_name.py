import inspect
import os
import posixpath
import re
import sys
import warnings
from inspect import Parameter
from os import path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Node, system_message
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import RSTStateMachine, Struct, state_classes
from docutils.statemachine import StringList
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.deprecation import (RemovedInSphinx60Warning, RemovedInSphinx70Warning,
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import INSTANCEATTR, Documenter
from sphinx.ext.autodoc.directive import DocumenterBridge, Options
from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autodoc.mock import mock
from sphinx.extension import Extension
from sphinx.locale import __
from sphinx.project import Project
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.registry import SphinxComponentRegistry
from sphinx.util import logging, rst
from sphinx.util.docutils import (NullReporter, SphinxDirective, SphinxRole, new_document,
from sphinx.util.inspect import signature_from_str
from sphinx.util.matching import Matcher
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
def _import_by_name(name: str, grouped_exception: bool=True) -> Tuple[Any, Any, str]:
    """Import a Python object given its full name."""
    errors: List[BaseException] = []
    try:
        name_parts = name.split('.')
        modname = '.'.join(name_parts[:-1])
        if modname:
            try:
                mod = import_module(modname)
                return (getattr(mod, name_parts[-1]), mod, modname)
            except (ImportError, IndexError, AttributeError) as exc:
                errors.append(exc.__cause__ or exc)
        last_j = 0
        modname = None
        for j in reversed(range(1, len(name_parts) + 1)):
            last_j = j
            modname = '.'.join(name_parts[:j])
            try:
                import_module(modname)
            except ImportError as exc:
                errors.append(exc.__cause__ or exc)
            if modname in sys.modules:
                break
        if last_j < len(name_parts):
            parent = None
            obj = sys.modules[modname]
            for obj_name in name_parts[last_j:]:
                parent = obj
                obj = getattr(obj, obj_name)
            return (obj, parent, modname)
        else:
            return (sys.modules[modname], None, modname)
    except (ValueError, ImportError, AttributeError, KeyError) as exc:
        errors.append(exc)
        if grouped_exception:
            raise ImportExceptionGroup('', errors)
        else:
            raise ImportError(*exc.args) from exc