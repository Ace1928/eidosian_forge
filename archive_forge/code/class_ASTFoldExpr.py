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
class ASTFoldExpr(ASTExpression):

    def __init__(self, leftExpr: ASTExpression, op: str, rightExpr: ASTExpression) -> None:
        assert leftExpr is not None or rightExpr is not None
        self.leftExpr = leftExpr
        self.op = op
        self.rightExpr = rightExpr

    def _stringify(self, transform: StringifyTransform) -> str:
        res = ['(']
        if self.leftExpr:
            res.append(transform(self.leftExpr))
            res.append(' ')
            res.append(self.op)
            res.append(' ')
        res.append('...')
        if self.rightExpr:
            res.append(' ')
            res.append(self.op)
            res.append(' ')
            res.append(transform(self.rightExpr))
        res.append(')')
        return ''.join(res)

    def get_id(self, version: int) -> str:
        assert version >= 3
        if version == 3:
            return str(self)
        res = []
        if self.leftExpr is None:
            res.append('fl')
        elif self.rightExpr is None:
            res.append('fr')
        else:
            res.append('fL')
        res.append(_id_operator_v2[self.op])
        if self.leftExpr:
            res.append(self.leftExpr.get_id(version))
        if self.rightExpr:
            res.append(self.rightExpr.get_id(version))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        signode += addnodes.desc_sig_punctuation('(', '(')
        if self.leftExpr:
            self.leftExpr.describe_signature(signode, mode, env, symbol)
            signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_operator(self.op, self.op)
            signode += addnodes.desc_sig_space()
        signode += addnodes.desc_sig_punctuation('...', '...')
        if self.rightExpr:
            signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_operator(self.op, self.op)
            signode += addnodes.desc_sig_space()
            self.rightExpr.describe_signature(signode, mode, env, symbol)
        signode += addnodes.desc_sig_punctuation(')', ')')