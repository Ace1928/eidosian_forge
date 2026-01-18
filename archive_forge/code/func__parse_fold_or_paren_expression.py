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
def _parse_fold_or_paren_expression(self) -> ASTExpression:
    if self.current_char != '(':
        return None
    self.pos += 1
    self.skip_ws()
    if self.skip_string_and_ws('...'):
        if not self.match(_fold_operator_re):
            self.fail("Expected fold operator after '...' in fold expression.")
        op = self.matched_text
        rightExpr = self._parse_cast_expression()
        if not self.skip_string(')'):
            self.fail("Expected ')' in end of fold expression.")
        return ASTFoldExpr(None, op, rightExpr)
    pos = self.pos
    try:
        self.skip_ws()
        leftExpr = self._parse_cast_expression()
        self.skip_ws()
        if not self.match(_fold_operator_re):
            self.fail('Expected fold operator after left expression in fold expression.')
        op = self.matched_text
        self.skip_ws()
        if not self.skip_string_and_ws('...'):
            self.fail("Expected '...' after fold operator in fold expression.")
    except DefinitionError as eFold:
        self.pos = pos
        try:
            res = self._parse_expression()
            self.skip_ws()
            if not self.skip_string(')'):
                self.fail("Expected ')' in end of parenthesized expression.")
        except DefinitionError as eExpr:
            raise self._make_multi_error([(eFold, 'If fold expression'), (eExpr, 'If parenthesized expression')], 'Error in fold expression or parenthesized expression.') from eExpr
        return ASTParenExpr(res)
    if self.skip_string(')'):
        return ASTFoldExpr(leftExpr, op, None)
    if not self.match(_fold_operator_re):
        self.fail("Expected fold operator or ')' after '...' in fold expression.")
    if op != self.matched_text:
        self.fail("Operators are different in binary fold: '%s' and '%s'." % (op, self.matched_text))
    rightExpr = self._parse_cast_expression()
    self.skip_ws()
    if not self.skip_string(')'):
        self.fail("Expected ')' to end binary fold expression.")
    return ASTFoldExpr(leftExpr, op, rightExpr)