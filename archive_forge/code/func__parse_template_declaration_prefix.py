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
def _parse_template_declaration_prefix(self, objectType: str) -> Optional[ASTTemplateDeclarationPrefix]:
    templates: List[Union[ASTTemplateParams, ASTTemplateIntroduction]] = []
    while 1:
        self.skip_ws()
        params: Union[ASTTemplateParams, ASTTemplateIntroduction] = None
        pos = self.pos
        if self.skip_word('template'):
            try:
                params = self._parse_template_parameter_list()
            except DefinitionError as e:
                if objectType == 'member' and len(templates) == 0:
                    return ASTTemplateDeclarationPrefix(None)
                else:
                    raise e
            if objectType == 'concept' and params.requiresClause is not None:
                self.fail('requires-clause not allowed for concept')
        else:
            params = self._parse_template_introduction()
            if not params:
                break
        if objectType == 'concept' and len(templates) > 0:
            self.pos = pos
            self.fail('More than 1 template parameter list for concept.')
        templates.append(params)
    if len(templates) == 0 and objectType == 'concept':
        self.fail('Missing template parameter list for concept.')
    if len(templates) == 0:
        return None
    else:
        return ASTTemplateDeclarationPrefix(templates)