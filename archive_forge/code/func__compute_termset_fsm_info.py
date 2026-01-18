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
def _compute_termset_fsm_info(self):
    """Collect and return information about terminal symbol sets and their FSMs.

        Terminal symbol sets (or "termsets") are ordered sequences of terminal
        symbols that are used by each parser state.  Associated with each is a
        collection of FSMs for each terminal and a single parse state FSM that is
        the union of each terminal's FSM.

        This constructs a list of tuples containing the termset, the set of
        parse states that use the termsets, parse state FSMs, and information
        mapping the components of the parse state FSMs to their terminal symbol
        FSMs.

        """
    context_lexer = get_contextual_lexer(self)
    termsets_to_fsms = {}
    termsets_to_parse_states: Dict[Tuple[str, ...], Set[ParseStateType]] = {}
    for parse_state, lexer in context_lexer.lexers.items():
        scanner = lexer.scanner
        key = tuple((term.name for term in scanner.terminals))
        termsets_to_fsms[key] = (scanner.fsm, scanner.fsms_to_trans_finals)
        termsets_to_parse_states.setdefault(key, set()).add(parse_state)
    self._termset_fsm_info = [(termset, frozenset(termsets_to_parse_states[termset]), fsm, fsms_to_trans_finals) for termset, (fsm, fsms_to_trans_finals) in termsets_to_fsms.items()]