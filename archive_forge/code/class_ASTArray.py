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
class ASTArray(ASTBase):

    def __init__(self, static: bool, const: bool, volatile: bool, restrict: bool, vla: bool, size: ASTExpression):
        self.static = static
        self.const = const
        self.volatile = volatile
        self.restrict = restrict
        self.vla = vla
        self.size = size
        if vla:
            assert size is None
        if size is not None:
            assert not vla

    def _stringify(self, transform: StringifyTransform) -> str:
        el = []
        if self.static:
            el.append('static')
        if self.restrict:
            el.append('restrict')
        if self.volatile:
            el.append('volatile')
        if self.const:
            el.append('const')
        if self.vla:
            return '[' + ' '.join(el) + '*]'
        elif self.size:
            el.append(transform(self.size))
        return '[' + ' '.join(el) + ']'

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        verify_description_mode(mode)
        signode += addnodes.desc_sig_punctuation('[', '[')
        addSpace = False

        def _add(signode: TextElement, text: str) -> bool:
            if addSpace:
                signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_keyword(text, text)
            return True
        if self.static:
            addSpace = _add(signode, 'static')
        if self.restrict:
            addSpace = _add(signode, 'restrict')
        if self.volatile:
            addSpace = _add(signode, 'volatile')
        if self.const:
            addSpace = _add(signode, 'const')
        if self.vla:
            signode += addnodes.desc_sig_punctuation('*', '*')
        elif self.size:
            if addSpace:
                signode += addnodes.desc_sig_space()
            self.size.describe_signature(signode, 'markType', env, symbol)
        signode += addnodes.desc_sig_punctuation(']', ']')