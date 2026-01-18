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
class PyFunction(PyObject):
    """Description of a function."""
    option_spec: OptionSpec = PyObject.option_spec.copy()
    option_spec.update({'async': directives.flag})

    def get_signature_prefix(self, sig: str) -> List[nodes.Node]:
        if 'async' in self.options:
            return [addnodes.desc_sig_keyword('', 'async'), addnodes.desc_sig_space()]
        else:
            return []

    def needs_arglist(self) -> bool:
        return True

    def add_target_and_index(self, name_cls: Tuple[str, str], sig: str, signode: desc_signature) -> None:
        super().add_target_and_index(name_cls, sig, signode)
        if 'noindexentry' not in self.options:
            modname = self.options.get('module', self.env.ref_context.get('py:module'))
            node_id = signode['ids'][0]
            name, cls = name_cls
            if modname:
                text = _('%s() (in module %s)') % (name, modname)
                self.indexnode['entries'].append(('single', text, node_id, '', None))
            else:
                text = '%s; %s()' % (pairindextypes['builtin'], name)
                self.indexnode['entries'].append(('pair', text, node_id, '', None))

    def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
        return None