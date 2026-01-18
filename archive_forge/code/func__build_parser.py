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
def _build_parser(self) -> 'PartialParsingFrontend':
    self._prepare_callbacks()
    _validate_frontend_args(self.options.parser, self.options.lexer)
    parser_conf = PartialParserConf(self.rules, self._callbacks, self.options.start, self.deterministic, self.use_value_stack)
    parser_type = self.options.parser
    lexer_type = self.options.lexer
    lexer_conf = self.lexer_conf
    assert isinstance(lexer_conf, LexerConf)
    assert isinstance(parser_conf, ParserConf)
    parser_conf.parser_type = parser_type
    self.lexer_conf.lexer_type = lexer_type
    return PartialParsingFrontend(lexer_conf, parser_conf, self.options)