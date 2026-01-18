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
def _parse_template_argument_list(self) -> ASTTemplateArgs:
    self.skip_ws()
    if not self.skip_string_and_ws('<'):
        return None
    if self.skip_string('>'):
        return ASTTemplateArgs([], False)
    prevErrors = []
    templateArgs: List[Union[ASTType, ASTTemplateArgConstant]] = []
    packExpansion = False
    while 1:
        pos = self.pos
        parsedComma = False
        parsedEnd = False
        try:
            type = self._parse_type(named=False)
            self.skip_ws()
            if self.skip_string_and_ws('...'):
                packExpansion = True
                parsedEnd = True
                if not self.skip_string('>'):
                    self.fail('Expected ">" after "..." in template argument list.')
            elif self.skip_string('>'):
                parsedEnd = True
            elif self.skip_string(','):
                parsedComma = True
            else:
                self.fail('Expected "...>", ">" or "," in template argument list.')
            templateArgs.append(type)
        except DefinitionError as e:
            prevErrors.append((e, 'If type argument'))
            self.pos = pos
            try:
                value = self._parse_constant_expression(inTemplate=True)
                self.skip_ws()
                if self.skip_string_and_ws('...'):
                    packExpansion = True
                    parsedEnd = True
                    if not self.skip_string('>'):
                        self.fail('Expected ">" after "..." in template argument list.')
                elif self.skip_string('>'):
                    parsedEnd = True
                elif self.skip_string(','):
                    parsedComma = True
                else:
                    self.fail('Expected "...>", ">" or "," in template argument list.')
                templateArgs.append(ASTTemplateArgConstant(value))
            except DefinitionError as e:
                self.pos = pos
                prevErrors.append((e, 'If non-type argument'))
                header = 'Error in parsing template argument list.'
                raise self._make_multi_error(prevErrors, header) from e
        if parsedEnd:
            assert not parsedComma
            break
        else:
            assert not packExpansion
    return ASTTemplateArgs(templateArgs, packExpansion)