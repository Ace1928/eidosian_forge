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
class PartialParsingFrontend(ParsingFrontend):

    def __init__(self, lexer_conf, parser_conf, options, parser=None):
        assert parser_conf.parser_type == 'lalr'
        options._plugins['LALR_Parser'] = PartialLALRParser
        options._plugins['BasicLexer'] = PartialBasicLexer
        options._plugins['ContextualLexer'] = PartialContextualLexer
        options._plugins['LexerThread'] = PartialLexerThread
        super().__init__(lexer_conf, parser_conf, options, parser=parser)
        if lexer_conf.postlex:
            self.lexer = PartialPostLexConnector(self.lexer.lexer, lexer_conf.postlex)
        self._termset_fsm_info = None
        self._symbols_to_states: Optional[Dict[str, Set[Tuple[ParseStateType, Action]]]] = None
        self._reverse_shifts: Optional[Dict[ParseStateType, Dict[str, Set[ParseStateType]]]] = None

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

    @property
    def termset_fsm_info(self):
        if self._termset_fsm_info is None:
            self._compute_termset_fsm_info()
        return self._termset_fsm_info

    @property
    def symbols_to_states(self):
        if self._symbols_to_states is None:
            self._compute_maps()
        return self._symbols_to_states

    @property
    def reverse_shifts(self):
        if self._reverse_shifts is None:
            self._compute_maps()
        return self._reverse_shifts