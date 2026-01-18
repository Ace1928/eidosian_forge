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
class PartialContextualLexer(ContextualLexer):

    def __init__(self, conf: 'LexerConf', states, always_accept=()):
        terminals = list(conf.terminals)
        terminals_by_name = conf.terminals_by_name
        trad_conf = copy(conf)
        trad_conf.terminals = terminals
        lexer_by_symbols: Dict = {}
        self.lexers = {}
        for state, accepts in states.items():
            key = frozenset(accepts)
            try:
                lexer = lexer_by_symbols[key]
            except KeyError:
                accepts = set(accepts) | set(conf.ignore) | set(always_accept)
                lexer_conf = copy(trad_conf)
                lexer_conf.terminals = [terminals_by_name[n] for n in accepts if n in terminals_by_name]
                lexer = PartialBasicLexer(lexer_conf)
                lexer_by_symbols[key] = lexer
            self.lexers[state] = lexer
        assert trad_conf.terminals is terminals
        self.root_lexer = PartialBasicLexer(trad_conf)

    def lex(self, lexer_state: LexerState, parser_state: Any) -> Iterator[Token]:
        try:
            while True:
                lexer = self.lexers[parser_state.position]
                yield lexer.next_token(lexer_state, parser_state)
        except EOFError:
            pass