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
class ASTUnaryOpExpr(ASTExpression):

    def __init__(self, op: str, expr: ASTExpression):
        self.op = op
        self.expr = expr

    def _stringify(self, transform: StringifyTransform) -> str:
        if self.op[0] in 'cn':
            return self.op + ' ' + transform(self.expr)
        else:
            return self.op + transform(self.expr)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        if self.op[0] in 'cn':
            signode += addnodes.desc_sig_keyword(self.op, self.op)
            signode += addnodes.desc_sig_space()
        else:
            signode += addnodes.desc_sig_operator(self.op, self.op)
        self.expr.describe_signature(signode, mode, env, symbol)