import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
class ASTTemplateKeyParamPackIdDefault(ASTTemplateParam):

    def __init__(self, key: str, identifier: ASTIdentifier, parameterPack: bool, default: ASTType) -> None:
        assert key
        if parameterPack:
            assert default is None
        self.key = key
        self.identifier = identifier
        self.parameterPack = parameterPack
        self.default = default

    def get_identifier(self) -> ASTIdentifier:
        return self.identifier

    def get_id(self, version: int) -> str:
        assert version >= 2
        res = []
        if self.parameterPack:
            res.append('Dp')
        else:
            res.append('0')
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = [self.key]
        if self.parameterPack:
            if self.identifier:
                res.append(' ')
            res.append('...')
        if self.identifier:
            if not self.parameterPack:
                res.append(' ')
            res.append(transform(self.identifier))
        if self.default:
            res.append(' = ')
            res.append(transform(self.default))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        signode += addnodes.desc_sig_keyword(self.key, self.key)
        if self.parameterPack:
            if self.identifier:
                signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_punctuation('...', '...')
        if self.identifier:
            if not self.parameterPack:
                signode += addnodes.desc_sig_space()
            self.identifier.describe_signature(signode, mode, env, '', '', symbol)
        if self.default:
            signode += addnodes.desc_sig_space()
            signode += addnodes.desc_sig_punctuation('=', '=')
            signode += addnodes.desc_sig_space()
            self.default.describe_signature(signode, 'markType', env, symbol)