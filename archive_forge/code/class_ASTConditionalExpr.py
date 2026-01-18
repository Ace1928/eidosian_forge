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
class ASTConditionalExpr(ASTExpression):

    def __init__(self, ifExpr: ASTExpression, thenExpr: ASTExpression, elseExpr: ASTExpression):
        self.ifExpr = ifExpr
        self.thenExpr = thenExpr
        self.elseExpr = elseExpr

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.ifExpr))
        res.append(' ? ')
        res.append(transform(self.thenExpr))
        res.append(' : ')
        res.append(transform(self.elseExpr))
        return ''.join(res)

    def get_id(self, version: int) -> str:
        assert version >= 2
        res = []
        res.append(_id_operator_v2['?'])
        res.append(self.ifExpr.get_id(version))
        res.append(self.thenExpr.get_id(version))
        res.append(self.elseExpr.get_id(version))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        self.ifExpr.describe_signature(signode, mode, env, symbol)
        signode += addnodes.desc_sig_space()
        signode += addnodes.desc_sig_operator('?', '?')
        signode += addnodes.desc_sig_space()
        self.thenExpr.describe_signature(signode, mode, env, symbol)
        signode += addnodes.desc_sig_space()
        signode += addnodes.desc_sig_operator(':', ':')
        signode += addnodes.desc_sig_space()
        self.elseExpr.describe_signature(signode, mode, env, symbol)