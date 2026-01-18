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
class ASTSizeofParamPack(ASTExpression):

    def __init__(self, identifier: ASTIdentifier):
        self.identifier = identifier

    def _stringify(self, transform: StringifyTransform) -> str:
        return 'sizeof...(' + transform(self.identifier) + ')'

    def get_id(self, version: int) -> str:
        return 'sZ' + self.identifier.get_id(version)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        signode += addnodes.desc_sig_keyword('sizeof', 'sizeof')
        signode += addnodes.desc_sig_punctuation('...', '...')
        signode += addnodes.desc_sig_punctuation('(', '(')
        self.identifier.describe_signature(signode, 'markType', env, symbol=symbol, prefix='', templateArgs='')
        signode += addnodes.desc_sig_punctuation(')', ')')