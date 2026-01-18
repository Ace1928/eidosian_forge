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
def _parse_declarator(self, named: Union[bool, str], paramMode: str, typed: bool=True) -> ASTDeclarator:
    if paramMode not in ('type', 'function'):
        raise Exception("Internal error, unknown paramMode '%s'." % paramMode)
    prevErrors = []
    self.skip_ws()
    if typed and self.skip_string('*'):
        self.skip_ws()
        restrict = False
        volatile = False
        const = False
        attrs = []
        while 1:
            if not restrict:
                restrict = self.skip_word_and_ws('restrict')
                if restrict:
                    continue
            if not volatile:
                volatile = self.skip_word_and_ws('volatile')
                if volatile:
                    continue
            if not const:
                const = self.skip_word_and_ws('const')
                if const:
                    continue
            attr = self._parse_attribute()
            if attr is not None:
                attrs.append(attr)
                continue
            break
        next = self._parse_declarator(named, paramMode, typed)
        return ASTDeclaratorPtr(next=next, restrict=restrict, volatile=volatile, const=const, attrs=ASTAttributeList(attrs))
    if typed and self.current_char == '(':
        pos = self.pos
        try:
            res = self._parse_declarator_name_suffix(named, paramMode, typed)
            return res
        except DefinitionError as exParamQual:
            msg = 'If declarator-id with parameters'
            if paramMode == 'function':
                msg += " (e.g., 'void f(int arg)')"
            prevErrors.append((exParamQual, msg))
            self.pos = pos
            try:
                assert self.current_char == '('
                self.skip_string('(')
                inner = self._parse_declarator(named, paramMode, typed)
                if not self.skip_string(')'):
                    self.fail('Expected \')\' in "( ptr-declarator )"')
                next = self._parse_declarator(named=False, paramMode='type', typed=typed)
                return ASTDeclaratorParen(inner=inner, next=next)
            except DefinitionError as exNoPtrParen:
                self.pos = pos
                msg = 'If parenthesis in noptr-declarator'
                if paramMode == 'function':
                    msg += " (e.g., 'void (*f(int arg))(double)')"
                prevErrors.append((exNoPtrParen, msg))
                header = 'Error in declarator'
                raise self._make_multi_error(prevErrors, header) from exNoPtrParen
    pos = self.pos
    try:
        return self._parse_declarator_name_suffix(named, paramMode, typed)
    except DefinitionError as e:
        self.pos = pos
        prevErrors.append((e, 'If declarator-id'))
        header = 'Error in declarator or parameters'
        raise self._make_multi_error(prevErrors, header) from e