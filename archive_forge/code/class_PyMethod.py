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
class PyMethod(PyObject):
    """Description of a method."""
    option_spec: OptionSpec = PyObject.option_spec.copy()
    option_spec.update({'abstractmethod': directives.flag, 'async': directives.flag, 'classmethod': directives.flag, 'final': directives.flag, 'property': directives.flag, 'staticmethod': directives.flag})

    def needs_arglist(self) -> bool:
        if 'property' in self.options:
            return False
        else:
            return True

    def get_signature_prefix(self, sig: str) -> List[nodes.Node]:
        prefix: List[nodes.Node] = []
        if 'final' in self.options:
            prefix.append(nodes.Text('final'))
            prefix.append(addnodes.desc_sig_space())
        if 'abstractmethod' in self.options:
            prefix.append(nodes.Text('abstract'))
            prefix.append(addnodes.desc_sig_space())
        if 'async' in self.options:
            prefix.append(nodes.Text('async'))
            prefix.append(addnodes.desc_sig_space())
        if 'classmethod' in self.options:
            prefix.append(nodes.Text('classmethod'))
            prefix.append(addnodes.desc_sig_space())
        if 'property' in self.options:
            logger.warning(_('Using the :property: flag with the py:method directiveis deprecated, use ".. py:property::" instead.'))
            prefix.append(nodes.Text('property'))
            prefix.append(addnodes.desc_sig_space())
        if 'staticmethod' in self.options:
            prefix.append(nodes.Text('static'))
            prefix.append(addnodes.desc_sig_space())
        return prefix

    def get_index_text(self, modname: str, name_cls: Tuple[str, str]) -> str:
        name, cls = name_cls
        try:
            clsname, methname = name.rsplit('.', 1)
            if modname and self.env.config.add_module_names:
                clsname = '.'.join([modname, clsname])
        except ValueError:
            if modname:
                return _('%s() (in module %s)') % (name, modname)
            else:
                return '%s()' % name
        if 'classmethod' in self.options:
            return _('%s() (%s class method)') % (methname, clsname)
        elif 'property' in self.options:
            return _('%s (%s property)') % (methname, clsname)
        elif 'staticmethod' in self.options:
            return _('%s() (%s static method)') % (methname, clsname)
        else:
            return _('%s() (%s method)') % (methname, clsname)