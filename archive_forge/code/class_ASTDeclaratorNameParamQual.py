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
class ASTDeclaratorNameParamQual(ASTDeclarator):

    def __init__(self, declId: ASTNestedName, arrayOps: List[ASTArray], paramQual: ASTParametersQualifiers) -> None:
        self.declId = declId
        self.arrayOps = arrayOps
        self.paramQual = paramQual

    @property
    def name(self) -> ASTNestedName:
        return self.declId

    @name.setter
    def name(self, name: ASTNestedName) -> None:
        self.declId = name

    @property
    def isPack(self) -> bool:
        return False

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.paramQual.function_params

    @property
    def trailingReturn(self) -> 'ASTType':
        return self.paramQual.trailingReturn

    def get_modifiers_id(self, version: int) -> str:
        if self.paramQual:
            return self.paramQual.get_modifiers_id(version)
        raise Exception('This should only be called on a function: %s' % self)

    def get_param_id(self, version: int) -> str:
        if self.paramQual:
            return self.paramQual.get_param_id(version)
        else:
            return ''

    def get_ptr_suffix_id(self, version: int) -> str:
        return ''.join((a.get_id(version) for a in self.arrayOps))

    def get_type_id(self, version: int, returnTypeId: str) -> str:
        assert version >= 2
        res = []
        res.append(self.get_ptr_suffix_id(version))
        if self.paramQual:
            res.append(self.get_modifiers_id(version))
            res.append('F')
            res.append(returnTypeId)
            res.append(self.get_param_id(version))
            res.append('E')
        else:
            res.append(returnTypeId)
        return ''.join(res)

    def require_space_after_declSpecs(self) -> bool:
        return self.declId is not None

    def is_function_type(self) -> bool:
        return self.paramQual is not None

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        if self.declId:
            res.append(transform(self.declId))
        for op in self.arrayOps:
            res.append(transform(op))
        if self.paramQual:
            res.append(transform(self.paramQual))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        verify_description_mode(mode)
        if self.declId:
            self.declId.describe_signature(signode, mode, env, symbol)
        for op in self.arrayOps:
            op.describe_signature(signode, mode, env, symbol)
        if self.paramQual:
            self.paramQual.describe_signature(signode, mode, env, symbol)