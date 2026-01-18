import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
class ASTIdentifier(ASTBaseBase):

    def __init__(self, identifier: str) -> None:
        assert identifier is not None
        assert len(identifier) != 0
        self.identifier = identifier

    def __eq__(self, other: Any) -> bool:
        return type(other) is ASTIdentifier and self.identifier == other.identifier

    def is_anon(self) -> bool:
        return self.identifier[0] == '@'

    def __str__(self) -> str:
        return self.identifier

    def get_display_string(self) -> str:
        return '[anonymous]' if self.is_anon() else self.identifier

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', prefix: str, symbol: 'Symbol') -> None:
        verify_description_mode(mode)
        if self.is_anon():
            node = addnodes.desc_sig_name(text='[anonymous]')
        else:
            node = addnodes.desc_sig_name(self.identifier, self.identifier)
        if mode == 'markType':
            targetText = prefix + self.identifier
            pnode = addnodes.pending_xref('', refdomain='c', reftype='identifier', reftarget=targetText, modname=None, classname=None)
            pnode['c:parent_key'] = symbol.get_lookup_key()
            pnode += node
            signode += pnode
        elif mode == 'lastIsName':
            nameNode = addnodes.desc_name()
            nameNode += node
            signode += nameNode
        elif mode == 'noneIsName':
            signode += node
        else:
            raise Exception('Unknown description mode: %s' % mode)