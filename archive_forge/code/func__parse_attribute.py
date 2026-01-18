import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
def _parse_attribute(self) -> Optional[ASTAttribute]:
    self.skip_ws()
    startPos = self.pos
    if self.skip_string_and_ws('['):
        if not self.skip_string('['):
            self.pos = startPos
        else:
            arg = self._parse_balanced_token_seq(end=[']'])
            if not self.skip_string_and_ws(']'):
                self.fail("Expected ']' in end of attribute.")
            if not self.skip_string_and_ws(']'):
                self.fail("Expected ']' in end of attribute after [[...]")
            return ASTCPPAttribute(arg)
    if self.skip_word_and_ws('__attribute__'):
        if not self.skip_string_and_ws('('):
            self.fail("Expected '(' after '__attribute__'.")
        if not self.skip_string_and_ws('('):
            self.fail("Expected '(' after '__attribute__('.")
        attrs = []
        while 1:
            if self.match(identifier_re):
                name = self.matched_text
                exprs = self._parse_paren_expression_list()
                attrs.append(ASTGnuAttribute(name, exprs))
            if self.skip_string_and_ws(','):
                continue
            elif self.skip_string_and_ws(')'):
                break
            else:
                self.fail("Expected identifier, ')', or ',' in __attribute__.")
        if not self.skip_string_and_ws(')'):
            self.fail("Expected ')' after '__attribute__((...)'")
        return ASTGnuAttributeList(attrs)
    for id in self.id_attributes:
        if self.skip_word_and_ws(id):
            return ASTIdAttribute(id)
    for id in self.paren_attributes:
        if not self.skip_string_and_ws(id):
            continue
        if not self.skip_string('('):
            self.fail("Expected '(' after user-defined paren-attribute.")
        arg = self._parse_balanced_token_seq(end=[')'])
        if not self.skip_string(')'):
            self.fail("Expected ')' to end user-defined paren-attribute.")
        return ASTParenAttribute(id, arg)
    return None