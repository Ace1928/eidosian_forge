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
class ASTDeclaratorPtr(ASTDeclarator):

    def __init__(self, next: ASTDeclarator, restrict: bool, volatile: bool, const: bool, attrs: ASTAttributeList) -> None:
        assert next
        self.next = next
        self.restrict = restrict
        self.volatile = volatile
        self.const = const
        self.attrs = attrs

    @property
    def name(self) -> ASTNestedName:
        return self.next.name

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.next.function_params

    def require_space_after_declSpecs(self) -> bool:
        return self.const or self.volatile or self.restrict or (len(self.attrs) > 0) or self.next.require_space_after_declSpecs()

    def _stringify(self, transform: StringifyTransform) -> str:
        res = ['*']
        res.append(transform(self.attrs))
        if len(self.attrs) != 0 and (self.restrict or self.volatile or self.const):
            res.append(' ')
        if self.restrict:
            res.append('restrict')
        if self.volatile:
            if self.restrict:
                res.append(' ')
            res.append('volatile')
        if self.const:
            if self.restrict or self.volatile:
                res.append(' ')
            res.append('const')
        if self.const or self.volatile or self.restrict or (len(self.attrs) > 0):
            if self.next.require_space_after_declSpecs():
                res.append(' ')
        res.append(transform(self.next))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        verify_description_mode(mode)
        signode += addnodes.desc_sig_punctuation('*', '*')
        self.attrs.describe_signature(signode)
        if len(self.attrs) != 0 and (self.restrict or self.volatile or self.const):
            signode += addnodes.desc_sig_space()

        def _add_anno(signode: TextElement, text: str) -> None:
            signode += addnodes.desc_sig_keyword(text, text)
        if self.restrict:
            _add_anno(signode, 'restrict')
        if self.volatile:
            if self.restrict:
                signode += addnodes.desc_sig_space()
            _add_anno(signode, 'volatile')
        if self.const:
            if self.restrict or self.volatile:
                signode += addnodes.desc_sig_space()
            _add_anno(signode, 'const')
        if self.const or self.volatile or self.restrict or (len(self.attrs) > 0):
            if self.next.require_space_after_declSpecs():
                signode += addnodes.desc_sig_space()
        self.next.describe_signature(signode, mode, env, symbol)