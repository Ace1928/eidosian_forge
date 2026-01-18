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
class PyXrefMixin:

    def make_xref(self, rolename: str, domain: str, target: str, innernode: Type[TextlikeNode]=nodes.emphasis, contnode: Node=None, env: BuildEnvironment=None, inliner: Inliner=None, location: Node=None) -> Node:
        result = super().make_xref(rolename, domain, target, innernode, contnode, env, inliner=None, location=None)
        if isinstance(result, pending_xref):
            result['refspecific'] = True
            result['py:module'] = env.ref_context.get('py:module')
            result['py:class'] = env.ref_context.get('py:class')
            reftype, reftarget, reftitle, _ = parse_reftarget(target)
            if reftarget != reftitle:
                result['reftype'] = reftype
                result['reftarget'] = reftarget
                result.clear()
                result += innernode(reftitle, reftitle)
            elif env.config.python_use_unqualified_type_names:
                children = result.children
                result.clear()
                shortname = target.split('.')[-1]
                textnode = innernode('', shortname)
                contnodes = [pending_xref_condition('', '', textnode, condition='resolved'), pending_xref_condition('', '', *children, condition='*')]
                result.extend(contnodes)
        return result

    def make_xrefs(self, rolename: str, domain: str, target: str, innernode: Type[TextlikeNode]=nodes.emphasis, contnode: Node=None, env: BuildEnvironment=None, inliner: Inliner=None, location: Node=None) -> List[Node]:
        delims = '(\\s*[\\[\\]\\(\\),](?:\\s*o[rf]\\s)?\\s*|\\s+o[rf]\\s+|\\s*\\|\\s*|\\.\\.\\.)'
        delims_re = re.compile(delims)
        sub_targets = re.split(delims, target)
        split_contnode = bool(contnode and contnode.astext() == target)
        in_literal = False
        results = []
        for sub_target in filter(None, sub_targets):
            if split_contnode:
                contnode = nodes.Text(sub_target)
            if in_literal or delims_re.match(sub_target):
                results.append(contnode or innernode(sub_target, sub_target))
            else:
                results.append(self.make_xref(rolename, domain, sub_target, innernode, contnode, env, inliner, location))
            if sub_target in ('Literal', 'typing.Literal', '~typing.Literal'):
                in_literal = True
        return results