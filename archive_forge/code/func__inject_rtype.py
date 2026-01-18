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
def _inject_rtype(type_hints: dict[str, Any], original_obj: Any, app: Sphinx, what: str, name: str, lines: list[str]) -> None:
    if inspect.isclass(original_obj) or inspect.isdatadescriptor(original_obj):
        return
    if what == 'method' and name.endswith('.__init__'):
        return
    if not app.config.typehints_document_rtype:
        return
    r = get_insert_index(app, lines)
    if r is None:
        return
    insert_index = r.insert_index
    if not app.config.typehints_use_rtype and r.found_return and (' -- ' in lines[insert_index]):
        return
    formatted_annotation = format_annotation(type_hints['return'], app.config)
    if r.found_param and insert_index < len(lines) and (lines[insert_index].strip() != ''):
        insert_index -= 1
    if insert_index == len(lines) and (not r.found_param):
        lines.append('')
        insert_index += 1
    if app.config.typehints_use_rtype or not r.found_return:
        line = f':rtype: {formatted_annotation}'
        lines.insert(insert_index, line)
        if r.found_directive:
            lines.insert(insert_index + 1, '')
    else:
        line = lines[insert_index]
        lines[insert_index] = f':return: {formatted_annotation} --{line[line.find(' '):]}'