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
def _compute_maps(self):
    """Compute state transition and symbols-to-states maps."""
    self._reverse_shifts = {}
    self._symbols_to_states = {}
    parse_table = self.parser.parser.parse_table
    for from_state, symbols_to_ops in parse_table.states.items():
        for symbol, op in symbols_to_ops.items():
            if op[0] == Shift:
                symbols_to_from_states = self._reverse_shifts.setdefault(op[1], {})
                symbols_to_from_states.setdefault(symbol, set()).add(from_state)
            self._symbols_to_states.setdefault(symbol, set()).add((from_state, op))