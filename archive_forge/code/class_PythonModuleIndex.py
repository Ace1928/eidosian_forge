import builtins
import inspect
import re
import sys
import typing
import warnings
from inspect import Parameter
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import Inliner
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref, pending_xref_condition
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, Index, IndexEntry, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import parse as ast_parse
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.inspect import signature_from_str
from sphinx.util.nodes import (find_pending_xref_condition, make_id, make_refnode,
from sphinx.util.typing import OptionSpec, TextlikeNode
class PythonModuleIndex(Index):
    """
    Index subclass to provide the Python module index.
    """
    name = 'modindex'
    localname = _('Python Module Index')
    shortname = _('modules')

    def generate(self, docnames: Iterable[str]=None) -> Tuple[List[Tuple[str, List[IndexEntry]]], bool]:
        content: Dict[str, List[IndexEntry]] = {}
        ignores: List[str] = self.domain.env.config['modindex_common_prefix']
        ignores = sorted(ignores, key=len, reverse=True)
        modules = sorted(self.domain.data['modules'].items(), key=lambda x: x[0].lower())
        prev_modname = ''
        num_toplevels = 0
        for modname, (docname, node_id, synopsis, platforms, deprecated) in modules:
            if docnames and docname not in docnames:
                continue
            for ignore in ignores:
                if modname.startswith(ignore):
                    modname = modname[len(ignore):]
                    stripped = ignore
                    break
            else:
                stripped = ''
            if not modname:
                modname, stripped = (stripped, '')
            entries = content.setdefault(modname[0].lower(), [])
            package = modname.split('.')[0]
            if package != modname:
                if prev_modname == package:
                    if entries:
                        last = entries[-1]
                        entries[-1] = IndexEntry(last[0], 1, last[2], last[3], last[4], last[5], last[6])
                elif not prev_modname.startswith(package):
                    entries.append(IndexEntry(stripped + package, 1, '', '', '', '', ''))
                subtype = 2
            else:
                num_toplevels += 1
                subtype = 0
            qualifier = _('Deprecated') if deprecated else ''
            entries.append(IndexEntry(stripped + modname, subtype, docname, node_id, platforms, qualifier, synopsis))
            prev_modname = modname
        collapse = len(modules) - num_toplevels < num_toplevels
        sorted_content = sorted(content.items())
        return (sorted_content, collapse)