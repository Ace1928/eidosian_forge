import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
def _parse_balanced_token_seq(self, end: List[str]) -> str:
    brackets = {'(': ')', '[': ']', '{': '}'}
    startPos = self.pos
    symbols: List[str] = []
    while not self.eof:
        if len(symbols) == 0 and self.current_char in end:
            break
        if self.current_char in brackets:
            symbols.append(brackets[self.current_char])
        elif len(symbols) > 0 and self.current_char == symbols[-1]:
            symbols.pop()
        elif self.current_char in ')]}':
            self.fail("Unexpected '%s' in balanced-token-seq." % self.current_char)
        self.pos += 1
    if self.eof:
        self.fail('Could not find end of balanced-token-seq starting at %d.' % startPos)
    return self.definition[startPos:self.pos]