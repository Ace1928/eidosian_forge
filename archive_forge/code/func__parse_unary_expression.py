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
def _parse_unary_expression(self) -> ASTExpression:
    self.skip_ws()
    for op in _expression_unary_ops:
        if op[0] in 'cn':
            res = self.skip_word(op)
        else:
            res = self.skip_string(op)
        if res:
            expr = self._parse_cast_expression()
            return ASTUnaryOpExpr(op, expr)
    if self.skip_word_and_ws('sizeof'):
        if self.skip_string_and_ws('('):
            typ = self._parse_type(named=False)
            self.skip_ws()
            if not self.skip_string(')'):
                self.fail("Expecting ')' to end 'sizeof'.")
            return ASTSizeofType(typ)
        expr = self._parse_unary_expression()
        return ASTSizeofExpr(expr)
    if self.skip_word_and_ws('alignof'):
        if not self.skip_string_and_ws('('):
            self.fail("Expecting '(' after 'alignof'.")
        typ = self._parse_type(named=False)
        self.skip_ws()
        if not self.skip_string(')'):
            self.fail("Expecting ')' to end 'alignof'.")
        return ASTAlignofExpr(typ)
    return self._parse_postfix_expression()