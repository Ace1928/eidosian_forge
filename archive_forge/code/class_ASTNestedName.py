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
class ASTNestedName(ASTBase):

    def __init__(self, names: List[ASTIdentifier], rooted: bool) -> None:
        assert len(names) > 0
        self.names = names
        self.rooted = rooted

    @property
    def name(self) -> 'ASTNestedName':
        return self

    def get_id(self, version: int) -> str:
        return '.'.join((str(n) for n in self.names))

    def _stringify(self, transform: StringifyTransform) -> str:
        res = '.'.join((transform(n) for n in self.names))
        if self.rooted:
            return '.' + res
        else:
            return res

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        verify_description_mode(mode)
        if mode == 'noneIsName':
            if self.rooted:
                raise AssertionError('Can this happen?')
                signode += nodes.Text('.')
            for i in range(len(self.names)):
                if i != 0:
                    raise AssertionError('Can this happen?')
                    signode += nodes.Text('.')
                n = self.names[i]
                n.describe_signature(signode, mode, env, '', symbol)
        elif mode == 'param':
            assert not self.rooted, str(self)
            assert len(self.names) == 1
            self.names[0].describe_signature(signode, 'noneIsName', env, '', symbol)
        elif mode in ('markType', 'lastIsName', 'markName'):
            prefix = ''
            first = True
            names = self.names[:-1] if mode == 'lastIsName' else self.names
            dest = signode
            if mode == 'lastIsName':
                dest = addnodes.desc_addname()
            if self.rooted:
                prefix += '.'
                if mode == 'lastIsName' and len(names) == 0:
                    signode += addnodes.desc_sig_punctuation('.', '.')
                else:
                    dest += addnodes.desc_sig_punctuation('.', '.')
            for i in range(len(names)):
                ident = names[i]
                if not first:
                    dest += addnodes.desc_sig_punctuation('.', '.')
                    prefix += '.'
                first = False
                txt_ident = str(ident)
                if txt_ident != '':
                    ident.describe_signature(dest, 'markType', env, prefix, symbol)
                prefix += txt_ident
            if mode == 'lastIsName':
                if len(self.names) > 1:
                    dest += addnodes.desc_sig_punctuation('.', '.')
                    signode += dest
                self.names[-1].describe_signature(signode, mode, env, '', symbol)
        else:
            raise Exception('Unknown description mode: %s' % mode)