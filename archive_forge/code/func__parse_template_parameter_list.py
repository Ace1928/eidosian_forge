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
def _parse_template_parameter_list(self) -> ASTTemplateParams:
    templateParams: List[ASTTemplateParam] = []
    self.skip_ws()
    if not self.skip_string('<'):
        self.fail("Expected '<' after 'template'")
    while 1:
        pos = self.pos
        err = None
        try:
            param = self._parse_template_parameter()
            templateParams.append(param)
        except DefinitionError as eParam:
            self.pos = pos
            err = eParam
        self.skip_ws()
        if self.skip_string('>'):
            requiresClause = self._parse_requires_clause()
            return ASTTemplateParams(templateParams, requiresClause)
        elif self.skip_string(','):
            continue
        else:
            header = 'Error in template parameter list.'
            errs = []
            if err:
                errs.append((err, 'If parameter'))
            try:
                self.fail('Expected "," or ">".')
            except DefinitionError as e:
                errs.append((e, 'If no parameter'))
            print(errs)
            raise self._make_multi_error(errs, header)