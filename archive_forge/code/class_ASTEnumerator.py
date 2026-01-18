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
class ASTEnumerator(ASTBase):

    def __init__(self, name: ASTNestedName, init: Optional[ASTInitializer], attrs: ASTAttributeList) -> None:
        self.name = name
        self.init = init
        self.attrs = attrs

    def get_id(self, version: int, objectType: str, symbol: 'Symbol') -> str:
        return symbol.get_full_nested_name().get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.name))
        if len(self.attrs) != 0:
            res.append(' ')
            res.append(transform(self.attrs))
        if self.init:
            res.append(transform(self.init))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        verify_description_mode(mode)
        self.name.describe_signature(signode, mode, env, symbol)
        if len(self.attrs) != 0:
            signode += addnodes.desc_sig_space()
            self.attrs.describe_signature(signode)
        if self.init:
            self.init.describe_signature(signode, 'markType', env, symbol)