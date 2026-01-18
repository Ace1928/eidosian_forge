from copy import copy, deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterator, Optional, Set, Tuple, Union
import interegular
from interegular.fsm import FSM
from interegular.patterns import Unsupported
from lark import Lark, Token
from lark.common import LexerConf, ParserConf
from lark.exceptions import LexError, UnexpectedInput
from lark.indenter import Indenter
from lark.lexer import (
from lark.parser_frontends import (
from lark.parsers.lalr_analysis import (
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser import LALR_Parser, ParseConf, ParserState, _Parser
from outlines.fsm.regex import (
class PartialIndenter(Indenter):
    """An `Indenter` that doesn't reset its state every time `process` is called."""

    def process(self, stream):
        return self._process(stream)

    def _process(self, stream):
        for token in stream:
            if token.type in self.OPEN_PAREN_types:
                self.paren_level += 1
            elif token.type in self.CLOSE_PAREN_types:
                self.paren_level -= 1
                if self.paren_level < 0:
                    raise UnexpectedToken(token, [])
            if token.type == self.NL_type:
                yield from self.handle_NL(token)
            else:
                yield token

    def accepts_token_type(self, token_type):
        if token_type in self.CLOSE_PAREN_types and self.paren_level - 1 < 0:
            return False
        return True

    def __copy__(self):
        res = type(self)()
        res.paren_level = self.paren_level
        res.indent_level = copy(self.indent_level)
        return res

    def __repr__(self):
        return f'{type(self).__name__}(paren_level={self.paren_level!r}, indent_level={self.indent_level!r})'