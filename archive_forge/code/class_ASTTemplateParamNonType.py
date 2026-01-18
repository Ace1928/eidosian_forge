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
class ASTTemplateParamNonType(ASTTemplateParam):

    def __init__(self, param: Union[ASTTypeWithInit, ASTTemplateParamConstrainedTypeWithInit], parameterPack: bool=False) -> None:
        assert param
        self.param = param
        self.parameterPack = parameterPack

    @property
    def name(self) -> ASTNestedName:
        id = self.get_identifier()
        return ASTNestedName([ASTNestedNameElement(id, None)], [False], rooted=False)

    @property
    def isPack(self) -> bool:
        return self.param.isPack or self.parameterPack

    def get_identifier(self) -> ASTIdentifier:
        name = self.param.name
        if name:
            assert len(name.names) == 1
            assert name.names[0].identOrOp
            assert not name.names[0].templateArgs
            res = name.names[0].identOrOp
            assert isinstance(res, ASTIdentifier)
            return res
        else:
            return None

    def get_id(self, version: int, objectType: str=None, symbol: 'Symbol'=None) -> str:
        assert version >= 2
        if symbol:
            return symbol.parent.declaration.get_id(version, prefixed=None)
        else:
            res = '_'
            if self.parameterPack:
                res += 'Dp'
            return res + self.param.get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = transform(self.param)
        if self.parameterPack:
            res += '...'
        return res

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        self.param.describe_signature(signode, mode, env, symbol)
        if self.parameterPack:
            signode += addnodes.desc_sig_punctuation('...', '...')