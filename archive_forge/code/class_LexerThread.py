from abc import abstractmethod, ABC
import re
from contextlib import suppress
from typing import (
from types import ModuleType
import warnings
from .utils import classify, get_regexp_width, Serialize, logger
from .exceptions import UnexpectedCharacters, LexError, UnexpectedToken
from .grammar import TOKEN_DEFAULT_PRIORITY
from copy import copy
class LexerThread:
    """A thread that ties a lexer instance and a lexer state, to be used by the parser
    """

    def __init__(self, lexer: 'Lexer', lexer_state: LexerState):
        self.lexer = lexer
        self.state = lexer_state

    @classmethod
    def from_text(cls, lexer: 'Lexer', text: str) -> 'LexerThread':
        return cls(lexer, LexerState(text))

    def lex(self, parser_state):
        return self.lexer.lex(self.state, parser_state)

    def __copy__(self):
        return type(self)(self.lexer, copy(self.state))
    _Token = Token