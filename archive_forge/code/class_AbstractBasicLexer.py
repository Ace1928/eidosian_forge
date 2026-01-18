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
class AbstractBasicLexer(Lexer):
    terminals_by_name: Dict[str, TerminalDef]

    @abstractmethod
    def __init__(self, conf: 'LexerConf', comparator=None) -> None:
        ...

    @abstractmethod
    def next_token(self, lex_state: LexerState, parser_state: Any=None) -> Token:
        ...

    def lex(self, state: LexerState, parser_state: Any) -> Iterator[Token]:
        with suppress(EOFError):
            while True:
                yield self.next_token(state, parser_state)