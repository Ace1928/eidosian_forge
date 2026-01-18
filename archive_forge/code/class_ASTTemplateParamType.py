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
class ASTTemplateParamType(ASTTemplateParam):

    def __init__(self, data: ASTTemplateKeyParamPackIdDefault) -> None:
        assert data
        self.data = data

    @property
    def name(self) -> ASTNestedName:
        id = self.get_identifier()
        return ASTNestedName([ASTNestedNameElement(id, None)], [False], rooted=False)

    @property
    def isPack(self) -> bool:
        return self.data.parameterPack

    def get_identifier(self) -> ASTIdentifier:
        return self.data.get_identifier()

    def get_id(self, version: int, objectType: str=None, symbol: 'Symbol'=None) -> str:
        assert version >= 2
        if symbol:
            return symbol.parent.declaration.get_id(version, prefixed=False)
        else:
            return self.data.get_id(version)

    def _stringify(self, transform: StringifyTransform) -> str:
        return transform(self.data)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        self.data.describe_signature(signode, mode, env, symbol)