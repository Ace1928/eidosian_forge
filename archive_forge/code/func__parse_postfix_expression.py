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
def _parse_postfix_expression(self) -> ASTPostfixExpr:
    prefix = self._parse_primary_expression()
    postFixes: List[ASTPostfixOp] = []
    while True:
        self.skip_ws()
        if self.skip_string_and_ws('['):
            expr = self._parse_expression()
            self.skip_ws()
            if not self.skip_string(']'):
                self.fail("Expected ']' in end of postfix expression.")
            postFixes.append(ASTPostfixArray(expr))
            continue
        if self.skip_string('->'):
            if self.skip_string('*'):
                self.pos -= 3
            else:
                name = self._parse_nested_name()
                postFixes.append(ASTPostfixMemberOfPointer(name))
                continue
        if self.skip_string('++'):
            postFixes.append(ASTPostfixInc())
            continue
        if self.skip_string('--'):
            postFixes.append(ASTPostfixDec())
            continue
        lst = self._parse_paren_expression_list()
        if lst is not None:
            postFixes.append(ASTPostfixCallExpr(lst))
            continue
        break
    return ASTPostfixExpr(prefix, postFixes)