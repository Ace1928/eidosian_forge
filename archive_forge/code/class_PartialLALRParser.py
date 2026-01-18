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
class PartialLALRParser(LALR_Parser):

    def __init__(self, parser_conf, debug=False, strict=False):
        analysis = LALR_Analyzer(parser_conf, debug=debug if not parser_conf.deterministic else True)
        analysis.compute_lalr()
        callbacks = parser_conf.callbacks
        self.parser_conf = parser_conf
        self._parse_table = analysis.parse_table
        if parser_conf.deterministic:
            old_to_new = {}

            def to_tuple(v):
                new = old_to_new.get(v)
                if new is None:
                    new = tuple(sorted(v, key=lambda y: str(y)))
                    old_to_new[v] = new
                return new
            enum = sorted(self._parse_table.states.keys(), key=lambda x: str(sorted(x, key=lambda y: str(y))))
            new_states = {}
            for s in enum:
                transitions = {term: op if op[0] is not Shift else (op[0], to_tuple(op[1])) for term, op in self._parse_table.states[s].items()}
                new_states[to_tuple(s)] = transitions
            self._parse_table = type(self._parse_table)(new_states, {k: to_tuple(v) for k, v in self._parse_table.start_states.items()}, {k: to_tuple(v) for k, v in self._parse_table.end_states.items()})
            if not debug:
                self._parse_table = IntParseTable.from_ParseTable(self._parse_table)
                self.states_to_rulesets = dict(zip(self._parse_table.states.keys(), new_states.keys()))
        self.parser = PartialParser(self._parse_table, callbacks, debug, use_value_stack=parser_conf.use_value_stack)

    @classmethod
    def deserialize(cls, data, memo, callbacks, debug=False):
        inst = cls.__new__(cls)
        inst._parse_table = ParseTable.deserialize(data, memo)
        inst.parser = PartialParser(inst._parse_table, callbacks, debug)
        return inst