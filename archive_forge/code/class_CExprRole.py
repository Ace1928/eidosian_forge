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
class CExprRole(SphinxRole):

    def __init__(self, asCode: bool) -> None:
        super().__init__()
        if asCode:
            self.class_type = 'c-expr'
        else:
            self.class_type = 'c-texpr'

    def run(self) -> Tuple[List[Node], List[system_message]]:
        text = self.text.replace('\n', ' ')
        parser = DefinitionParser(text, location=self.get_location(), config=self.env.config)
        try:
            ast = parser.parse_expression()
        except DefinitionError as ex:
            logger.warning('Unparseable C expression: %r\n%s', text, ex, location=self.get_location())
            return ([addnodes.desc_inline('c', text, text, classes=[self.class_type])], [])
        parentSymbol = self.env.temp_data.get('c:parent_symbol', None)
        if parentSymbol is None:
            parentSymbol = self.env.domaindata['c']['root_symbol']
        signode = addnodes.desc_inline('c', classes=[self.class_type])
        ast.describe_signature(signode, 'markType', self.env, parentSymbol)
        return ([signode], [])