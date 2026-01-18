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
def _parse_initializer(self, outer: str=None, allowFallback: bool=True) -> ASTInitializer:
    self.skip_ws()
    if outer == 'member' and False:
        bracedInit = self._parse_braced_init_list()
        if bracedInit is not None:
            return ASTInitializer(bracedInit, hasAssign=False)
    if not self.skip_string('='):
        return None
    bracedInit = self._parse_braced_init_list()
    if bracedInit is not None:
        return ASTInitializer(bracedInit)
    if outer == 'member':
        fallbackEnd: List[str] = []
    elif outer is None:
        fallbackEnd = [',', ')']
    else:
        self.fail("Internal error, initializer for outer '%s' not implemented." % outer)

    def parser():
        return self._parse_assignment_expression()
    value = self._parse_expression_fallback(fallbackEnd, parser, allow=allowFallback)
    return ASTInitializer(value)