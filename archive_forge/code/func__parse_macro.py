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
def _parse_macro(self) -> ASTMacro:
    self.skip_ws()
    ident = self._parse_nested_name()
    if ident is None:
        self.fail('Expected identifier in macro definition.')
    self.skip_ws()
    if not self.skip_string_and_ws('('):
        return ASTMacro(ident, None)
    if self.skip_string(')'):
        return ASTMacro(ident, [])
    args = []
    while 1:
        self.skip_ws()
        if self.skip_string('...'):
            args.append(ASTMacroParameter(None, True))
            self.skip_ws()
            if not self.skip_string(')'):
                self.fail('Expected ")" after "..." in macro parameters.')
            break
        if not self.match(identifier_re):
            self.fail('Expected identifier in macro parameters.')
        nn = ASTNestedName([ASTIdentifier(self.matched_text)], rooted=False)
        self.skip_ws()
        if self.skip_string_and_ws('...'):
            args.append(ASTMacroParameter(nn, False, True))
            self.skip_ws()
            if not self.skip_string(')'):
                self.fail('Expected ")" after "..." in macro parameters.')
            break
        args.append(ASTMacroParameter(nn))
        if self.skip_string_and_ws(','):
            continue
        elif self.skip_string_and_ws(')'):
            break
        else:
            self.fail("Expected identifier, ')', or ',' in macro parameter list.")
    return ASTMacro(ident, args)