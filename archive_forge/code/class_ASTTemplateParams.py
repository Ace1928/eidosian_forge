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
class ASTTemplateParams(ASTBase):

    def __init__(self, params: List[ASTTemplateParam], requiresClause: Optional['ASTRequiresClause']) -> None:
        assert params is not None
        self.params = params
        self.requiresClause = requiresClause

    def get_id(self, version: int, excludeRequires: bool=False) -> str:
        assert version >= 2
        res = []
        res.append('I')
        for param in self.params:
            res.append(param.get_id(version))
        res.append('E')
        if not excludeRequires and self.requiresClause:
            res.append('IQ')
            res.append(self.requiresClause.expr.get_id(version))
            res.append('E')
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append('template<')
        res.append(', '.join((transform(a) for a in self.params)))
        res.append('> ')
        if self.requiresClause is not None:
            res.append(transform(self.requiresClause))
            res.append(' ')
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        signode += addnodes.desc_sig_keyword('template', 'template')
        signode += addnodes.desc_sig_punctuation('<', '<')
        first = True
        for param in self.params:
            if not first:
                signode += addnodes.desc_sig_punctuation(',', ',')
                signode += addnodes.desc_sig_space()
            first = False
            param.describe_signature(signode, mode, env, symbol)
        signode += addnodes.desc_sig_punctuation('>', '>')
        if self.requiresClause is not None:
            signode += addnodes.desc_sig_space()
            self.requiresClause.describe_signature(signode, mode, env, symbol)

    def describe_signature_as_introducer(self, parentNode: desc_signature, mode: str, env: 'BuildEnvironment', symbol: 'Symbol', lineSpec: bool) -> None:

        def makeLine(parentNode: desc_signature) -> addnodes.desc_signature_line:
            signode = addnodes.desc_signature_line()
            parentNode += signode
            signode.sphinx_line_type = 'templateParams'
            return signode
        lineNode = makeLine(parentNode)
        lineNode += addnodes.desc_sig_keyword('template', 'template')
        lineNode += addnodes.desc_sig_punctuation('<', '<')
        first = True
        for param in self.params:
            if not first:
                lineNode += addnodes.desc_sig_punctuation(',', ',')
                lineNode += addnodes.desc_sig_space()
            first = False
            if lineSpec:
                lineNode = makeLine(parentNode)
            param.describe_signature(lineNode, mode, env, symbol)
        if lineSpec and (not first):
            lineNode = makeLine(parentNode)
        lineNode += addnodes.desc_sig_punctuation('>', '>')
        if self.requiresClause:
            reqNode = addnodes.desc_signature_line()
            reqNode.sphinx_line_type = 'requiresClause'
            parentNode += reqNode
            self.requiresClause.describe_signature(reqNode, 'markType', env, symbol)