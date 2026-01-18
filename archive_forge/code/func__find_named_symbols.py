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
def _find_named_symbols(self, ident: ASTIdentifier, matchSelf: bool, recurseInAnon: bool, searchInSiblings: bool) -> Iterator['Symbol']:
    if Symbol.debug_lookup:
        Symbol.debug_indent += 1
        Symbol.debug_print('_find_named_symbols:')
        Symbol.debug_indent += 1
        Symbol.debug_print('self:')
        print(self.to_string(Symbol.debug_indent + 1), end='')
        Symbol.debug_print('ident:            ', ident)
        Symbol.debug_print('matchSelf:        ', matchSelf)
        Symbol.debug_print('recurseInAnon:    ', recurseInAnon)
        Symbol.debug_print('searchInSiblings: ', searchInSiblings)

    def candidates() -> Generator['Symbol', None, None]:
        s = self
        if Symbol.debug_lookup:
            Symbol.debug_print('searching in self:')
            print(s.to_string(Symbol.debug_indent + 1), end='')
        while True:
            if matchSelf:
                yield s
            if recurseInAnon:
                yield from s.children_recurse_anon
            else:
                yield from s._children
            if s.siblingAbove is None:
                break
            s = s.siblingAbove
            if Symbol.debug_lookup:
                Symbol.debug_print('searching in sibling:')
                print(s.to_string(Symbol.debug_indent + 1), end='')
    for s in candidates():
        if Symbol.debug_lookup:
            Symbol.debug_print('candidate:')
            print(s.to_string(Symbol.debug_indent + 1), end='')
        if s.ident == ident:
            if Symbol.debug_lookup:
                Symbol.debug_indent += 1
                Symbol.debug_print('matches')
                Symbol.debug_indent -= 3
            yield s
            if Symbol.debug_lookup:
                Symbol.debug_indent += 2
    if Symbol.debug_lookup:
        Symbol.debug_indent -= 2