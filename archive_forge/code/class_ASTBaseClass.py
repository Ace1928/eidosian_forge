import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
class ASTBaseClass(ASTBase):

    def __init__(self, name: ASTNestedName, visibility: str, virtual: bool, pack: bool) -> None:
        self.name = name
        self.visibility = visibility
        self.virtual = virtual
        self.pack = pack

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.visibility is not None:
            res.append(self.visibility)
            res.append(' ')
        if self.virtual:
            res.append('virtual ')
        res.append(transform(self.name))
        if self.pack:
            res.append('...')
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        verify_description_mode(mode)
        if self.visibility is not None:
            signode += addnodes.desc_sig_keyword(self.visibility, self.visibility)
            signode += addnodes.desc_sig_space()
        if self.virtual:
            signode += addnodes.desc_sig_keyword('virtual', 'virtual')
            signode += addnodes.desc_sig_space()
        self.name.describe_signature(signode, 'markType', env, symbol=symbol)
        if self.pack:
            signode += addnodes.desc_sig_punctuation('...', '...')