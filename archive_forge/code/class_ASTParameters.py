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
class ASTParameters(ASTBase):

    def __init__(self, args: List[ASTFunctionParameter], attrs: ASTAttributeList) -> None:
        self.args = args
        self.attrs = attrs

    @property
    def function_params(self) -> List[ASTFunctionParameter]:
        return self.args

    def _stringify(self, transform: StringifyTransform) -> str:
        res = []
        res.append('(')
        first = True
        for a in self.args:
            if not first:
                res.append(', ')
            first = False
            res.append(str(a))
        res.append(')')
        if len(self.attrs) != 0:
            res.append(' ')
            res.append(transform(self.attrs))
        return ''.join(res)

    def describe_signature(self, signode: TextElement, mode: str, env: 'BuildEnvironment', symbol: 'Symbol') -> None:
        verify_description_mode(mode)
        if mode == 'lastIsName':
            paramlist = addnodes.desc_parameterlist()
            for arg in self.args:
                param = addnodes.desc_parameter('', '', noemph=True)
                arg.describe_signature(param, 'param', env, symbol=symbol)
                paramlist += param
            signode += paramlist
        else:
            signode += addnodes.desc_sig_punctuation('(', '(')
            first = True
            for arg in self.args:
                if not first:
                    signode += addnodes.desc_sig_punctuation(',', ',')
                    signode += addnodes.desc_sig_space()
                first = False
                arg.describe_signature(signode, 'markType', env, symbol=symbol)
            signode += addnodes.desc_sig_punctuation(')', ')')
        if len(self.attrs) != 0:
            signode += addnodes.desc_sig_space()
            self.attrs.describe_signature(signode)