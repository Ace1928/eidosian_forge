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
def _parse_template_introduction(self) -> ASTTemplateIntroduction:
    pos = self.pos
    try:
        concept = self._parse_nested_name()
    except Exception:
        self.pos = pos
        return None
    self.skip_ws()
    if not self.skip_string('{'):
        self.pos = pos
        return None
    params = []
    while 1:
        self.skip_ws()
        parameterPack = self.skip_string('...')
        self.skip_ws()
        if not self.match(identifier_re):
            self.fail('Expected identifier in template introduction list.')
        txt_identifier = self.matched_text
        if txt_identifier in _keywords:
            self.fail('Expected identifier in template introduction list, got keyword: %s' % txt_identifier)
        identifier = ASTIdentifier(txt_identifier)
        params.append(ASTTemplateIntroductionParameter(identifier, parameterPack))
        self.skip_ws()
        if self.skip_string('}'):
            break
        elif self.skip_string(','):
            continue
        else:
            self.fail('Error in template introduction list. Expected ",", or "}".')
    return ASTTemplateIntroduction(concept, params)