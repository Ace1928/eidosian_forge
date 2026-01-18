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
def _parse_requires_clause(self) -> Optional[ASTRequiresClause]:
    self.skip_ws()
    if not self.skip_word('requires'):
        return None

    def parse_and_expr(self: DefinitionParser) -> ASTExpression:
        andExprs = []
        ops = []
        andExprs.append(self._parse_primary_expression())
        while True:
            self.skip_ws()
            oneMore = False
            if self.skip_string('&&'):
                oneMore = True
                ops.append('&&')
            elif self.skip_word('and'):
                oneMore = True
                ops.append('and')
            if not oneMore:
                break
            andExprs.append(self._parse_primary_expression())
        if len(andExprs) == 1:
            return andExprs[0]
        else:
            return ASTBinOpExpr(andExprs, ops)
    orExprs = []
    ops = []
    orExprs.append(parse_and_expr(self))
    while True:
        self.skip_ws()
        oneMore = False
        if self.skip_string('||'):
            oneMore = True
            ops.append('||')
        elif self.skip_word('or'):
            oneMore = True
            ops.append('or')
        if not oneMore:
            break
        orExprs.append(parse_and_expr(self))
    if len(orExprs) == 1:
        return ASTRequiresClause(orExprs[0])
    else:
        return ASTRequiresClause(ASTBinOpExpr(orExprs, ops))