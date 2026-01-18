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
class PartialParserConf(ParserConf):
    __serialize_fields__ = ('rules', 'start', 'parser_type', 'deterministic', 'use_value_stack')

    def __init__(self, rules, callbacks, start, deterministic, use_value_stack):
        super().__init__(rules, callbacks, start)
        self.deterministic = deterministic
        self.use_value_stack = use_value_stack