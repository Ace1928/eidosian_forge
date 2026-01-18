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
def _parse_cast_expression(self) -> ASTExpression:
    pos = self.pos
    self.skip_ws()
    if self.skip_string('('):
        try:
            typ = self._parse_type(False)
            if not self.skip_string(')'):
                self.fail("Expected ')' in cast expression.")
            expr = self._parse_cast_expression()
            return ASTCastExpr(typ, expr)
        except DefinitionError as exCast:
            self.pos = pos
            try:
                return self._parse_unary_expression()
            except DefinitionError as exUnary:
                errs = []
                errs.append((exCast, 'If type cast expression'))
                errs.append((exUnary, 'If unary expression'))
                raise self._make_multi_error(errs, 'Error in cast expression.') from exUnary
    else:
        return self._parse_unary_expression()