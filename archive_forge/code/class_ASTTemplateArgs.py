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
class ASTTemplateArgs(ASTBase):

    def __init__(self, args: List[Union['ASTType', ASTTemplateArgConstant]], packExpansion: bool) -> None:
        assert args is not None
        self.args = args
        self.packExpansion = packExpansion

    def get_id(self, version: int) -> str:
        if version == 1:
            res = []
            res.append(':')
            res.append('.'.join((a.get_id(version) for a in self.args)))
            res.append(':')
            return ''.join(res)
        res = []
        res.append('I')
        if len(self.args) > 0:
            for a in self.args[:-1]:
                res.append(a.get_id(version))
            if self.packExpansion:
                res.append('J')
            res.append(self.args[-1].get_id(version))
            if self.packExpansion:
                res.append('E')
        res.append('E')
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = ', '.join((transform(a) for a in self.args))
        if self.packExpansion:
            res += '...'
        return '<' + res + '>'

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        verify_description_mode(mode)
        signode += addnodes.desc_sig_punctuation('<', '<')
        first = True
        for a in self.args:
            if not first:
                signode += addnodes.desc_sig_punctuation(',', ',')
                signode += addnodes.desc_sig_space()
            first = False
            a.describe_signature(signode, 'markType', env, symbol=symbol)
        if self.packExpansion:
            signode += addnodes.desc_sig_punctuation('...', '...')
        signode += addnodes.desc_sig_punctuation('>', '>')