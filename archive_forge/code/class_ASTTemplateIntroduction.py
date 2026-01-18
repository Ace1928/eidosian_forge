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
class ASTTemplateIntroduction(ASTBase):

    def __init__(self, concept: ASTNestedName, params: List[ASTTemplateIntroductionParameter]) -> None:
        assert len(params) > 0
        self.concept = concept
        self.params = params

    def get_id(self, version: int) -> str:
        assert version >= 2
        res = []
        res.append('I')
        for param in self.params:
            res.append(param.get_id(version))
        res.append('E')
        res.append('X')
        res.append(self.concept.get_id(version))
        res.append('I')
        for param in self.params:
            res.append(param.get_id_as_arg(version))
        res.append('E')
        res.append('E')
        return ''.join(res)

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append(transform(self.concept))
        res.append('{')
        res.append(', '.join((transform(param) for param in self.params)))
        res.append('} ')
        return ''.join(res)

    def describe_signature_as_introducer(self, parentNode: desc_signature, mode: str, env: 'BuildEnvironment', symbol: 'Symbol', lineSpec: bool) -> None:
        signode = addnodes.desc_signature_line()
        parentNode += signode
        signode.sphinx_line_type = 'templateIntroduction'
        self.concept.describe_signature(signode, 'markType', env, symbol)
        signode += addnodes.desc_sig_punctuation('{', '{')
        first = True
        for param in self.params:
            if not first:
                signode += addnodes.desc_sig_punctuation(',', ',')
                signode += addnodes.desc_sig_space()
            first = False
            param.describe_signature(signode, mode, env, symbol)
        signode += addnodes.desc_sig_punctuation('}', '}')