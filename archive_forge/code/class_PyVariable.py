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
class PyVariable(PyObject):
    """Description of a variable."""
    option_spec: OptionSpec = PyObject.option_spec.copy()
    option_spec.update({'type': directives.unchanged, 'value': directives.unchanged})

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        fullname, prefix = super().handle_signature(sig, signode)
        typ = self.options.get('type')
        if typ:
            annotations = _parse_annotation(typ, self.env)
            signode += addnodes.desc_annotation(typ, '', addnodes.desc_sig_punctuation('', ':'), addnodes.desc_sig_space(), *annotations)
        value = self.options.get('value')
        if value:
            signode += addnodes.desc_annotation(value, '', addnodes.desc_sig_space(), addnodes.desc_sig_punctuation('', '='), addnodes.desc_sig_space(), nodes.Text(value))
        return (fullname, prefix)

    def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
        name, cls = name_cls
        if modname:
            return _('%s (in module %s)') % (name, modname)
        else:
            return _('%s (built-in variable)') % name