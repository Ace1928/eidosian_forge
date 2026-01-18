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
def _add_symbols(self, nestedName: ASTNestedName, declaration: ASTDeclaration, docname: str, line: int) -> 'Symbol':
    if Symbol.debug_lookup:
        Symbol.debug_indent += 1
        Symbol.debug_print('_add_symbols:')
        Symbol.debug_indent += 1
        Symbol.debug_print('nn:       ', nestedName)
        Symbol.debug_print('decl:     ', declaration)
        Symbol.debug_print('location: {}:{}'.format(docname, line))

    def onMissingQualifiedSymbol(parentSymbol: 'Symbol', ident: ASTIdentifier) -> 'Symbol':
        if Symbol.debug_lookup:
            Symbol.debug_indent += 1
            Symbol.debug_print('_add_symbols, onMissingQualifiedSymbol:')
            Symbol.debug_indent += 1
            Symbol.debug_print('ident: ', ident)
            Symbol.debug_indent -= 2
        return Symbol(parent=parentSymbol, ident=ident, declaration=None, docname=None, line=None)
    lookupResult = self._symbol_lookup(nestedName, onMissingQualifiedSymbol, ancestorLookupType=None, matchSelf=False, recurseInAnon=False, searchInSiblings=False)
    assert lookupResult is not None
    symbols = list(lookupResult.symbols)
    if len(symbols) == 0:
        if Symbol.debug_lookup:
            Symbol.debug_print('_add_symbols, result, no symbol:')
            Symbol.debug_indent += 1
            Symbol.debug_print('ident:       ', lookupResult.ident)
            Symbol.debug_print('declaration: ', declaration)
            Symbol.debug_print('location:    {}:{}'.format(docname, line))
            Symbol.debug_indent -= 1
        symbol = Symbol(parent=lookupResult.parentSymbol, ident=lookupResult.ident, declaration=declaration, docname=docname, line=line)
        if Symbol.debug_lookup:
            Symbol.debug_indent -= 2
        return symbol
    if Symbol.debug_lookup:
        Symbol.debug_print('_add_symbols, result, symbols:')
        Symbol.debug_indent += 1
        Symbol.debug_print('number symbols:', len(symbols))
        Symbol.debug_indent -= 1
    if not declaration:
        if Symbol.debug_lookup:
            Symbol.debug_print('no declaration')
            Symbol.debug_indent -= 2
        return symbols[0]
    noDecl = []
    withDecl = []
    dupDecl = []
    for s in symbols:
        if s.declaration is None:
            noDecl.append(s)
        elif s.isRedeclaration:
            dupDecl.append(s)
        else:
            withDecl.append(s)
    if Symbol.debug_lookup:
        Symbol.debug_print('#noDecl:  ', len(noDecl))
        Symbol.debug_print('#withDecl:', len(withDecl))
        Symbol.debug_print('#dupDecl: ', len(dupDecl))

    def makeCandSymbol() -> 'Symbol':
        if Symbol.debug_lookup:
            Symbol.debug_print('begin: creating candidate symbol')
        symbol = Symbol(parent=lookupResult.parentSymbol, ident=lookupResult.ident, declaration=declaration, docname=docname, line=line)
        if Symbol.debug_lookup:
            Symbol.debug_print('end:   creating candidate symbol')
        return symbol
    if len(withDecl) == 0:
        candSymbol = None
    else:
        candSymbol = makeCandSymbol()

        def handleDuplicateDeclaration(symbol: 'Symbol', candSymbol: 'Symbol') -> None:
            if Symbol.debug_lookup:
                Symbol.debug_indent += 1
                Symbol.debug_print('redeclaration')
                Symbol.debug_indent -= 1
                Symbol.debug_indent -= 2
            candSymbol.isRedeclaration = True
            raise _DuplicateSymbolError(symbol, declaration)
        if declaration.objectType != 'function':
            assert len(withDecl) <= 1
            handleDuplicateDeclaration(withDecl[0], candSymbol)
        candId = declaration.get_newest_id()
        if Symbol.debug_lookup:
            Symbol.debug_print('candId:', candId)
        for symbol in withDecl:
            oldId = symbol.declaration.get_newest_id()
            if Symbol.debug_lookup:
                Symbol.debug_print('oldId: ', oldId)
            if candId == oldId:
                handleDuplicateDeclaration(symbol, candSymbol)
    if len(noDecl) == 0:
        if Symbol.debug_lookup:
            Symbol.debug_print('no match, no empty, candSybmol is not None?:', candSymbol is not None)
            Symbol.debug_indent -= 2
        if candSymbol is not None:
            return candSymbol
        else:
            return makeCandSymbol()
    else:
        if Symbol.debug_lookup:
            Symbol.debug_print('no match, but fill an empty declaration, candSybmol is not None?:', candSymbol is not None)
            Symbol.debug_indent -= 2
        if candSymbol is not None:
            candSymbol.remove()
        symbol = noDecl[0]
        symbol._fill_empty(declaration, docname, line)
        return symbol