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
def _parse_initializer_list(self, name: str, open: str, close: str) -> Tuple[List[ASTExpression], bool]:
    self.skip_ws()
    if not self.skip_string_and_ws(open):
        return (None, None)
    if self.skip_string(close):
        return ([], False)
    exprs = []
    trailingComma = False
    while True:
        self.skip_ws()
        expr = self._parse_expression()
        self.skip_ws()
        exprs.append(expr)
        self.skip_ws()
        if self.skip_string(close):
            break
        if not self.skip_string_and_ws(','):
            self.fail("Error in %s, expected ',' or '%s'." % (name, close))
        if self.current_char == close and close == '}':
            self.pos += 1
            trailingComma = True
            break
    return (exprs, trailingComma)