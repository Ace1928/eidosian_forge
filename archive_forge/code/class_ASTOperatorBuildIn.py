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
class ASTOperatorBuildIn(ASTOperator):

    def __init__(self, op: str) -> None:
        self.op = op

    def get_id(self, version: int) -> str:
        if version == 1:
            ids = _id_operator_v1
            if self.op not in ids:
                raise NoOldIdError()
        else:
            ids = _id_operator_v2
        if self.op not in ids:
            raise Exception('Internal error: Built-in operator "%s" can not be mapped to an id.' % self.op)
        return ids[self.op]

    def _stringify(self, transform: StringifyTransform) -> str:
        if self.op in ('new', 'new[]', 'delete', 'delete[]') or self.op[0] in 'abcnox':
            return 'operator ' + self.op
        else:
            return 'operator' + self.op

    def _describe_identifier(self, signode: TextElement, identnode: TextElement, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        signode += addnodes.desc_sig_keyword('operator', 'operator')
        if self.op in ('new', 'new[]', 'delete', 'delete[]') or self.op[0] in 'abcnox':
            signode += addnodes.desc_sig_space()
        identnode += addnodes.desc_sig_operator(self.op, self.op)