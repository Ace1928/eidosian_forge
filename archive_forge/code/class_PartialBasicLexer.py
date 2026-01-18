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
class PartialBasicLexer(BasicLexer):

    def __init__(self, conf: 'LexerConf'):
        super().__init__(conf)
        self._build_scanner()

    def _build_scanner(self):
        terminals, self.callback = _create_unless(self.terminals, self.g_regex_flags, self.re, self.use_bytes)
        assert not self.user_callbacks
        for terminal_name, callback in self.callback.items():
            terminal = self.terminals_by_name[terminal_name]
            for sub_terminal in callback.scanner.terminals:
                self.terminals.remove(sub_terminal)
                idx = self.terminals.index(terminal)
                self.terminals.insert(idx, sub_terminal)
        self._scanner = PartialScanner(self.terminals, self.g_regex_flags, self.re, self.use_bytes)

    def match(self, text, pos, last_fsm_state_seq=None):
        return self.scanner.match(text, pos, last_fsm_state_seq)

    def next_token(self, lex_state: LexerState, parser_state: Any=None) -> Token:
        last_token = lex_state.last_token
        last_fsm_state_seq = None
        if last_token and last_token.type == 'partial':
            last_fsm_state_seq = last_token.value.fsm_state_seq
        line_ctr = lex_state.line_ctr
        end_pos = line_ctr.char_pos + (len(last_fsm_state_seq) - 1 if last_fsm_state_seq else 0)
        while end_pos < len(lex_state.text):
            res = self.match(lex_state.text, line_ctr.char_pos, last_fsm_state_seq)
            if not res:
                if not last_fsm_state_seq or last_fsm_state_seq[-1] not in self.scanner.fsm.finals:
                    allowed = self.scanner.allowed_types - self.ignore_types
                    if not allowed:
                        allowed = {'<END-OF-FILE>'}
                    raise UnexpectedCharacters(lex_state.text, line_ctr.char_pos, line_ctr.line, line_ctr.column, allowed=allowed, token_history=lex_state.last_token and [lex_state.last_token], state=parser_state, terminals_by_name=self.terminals_by_name)
                fsm_state_seq = last_token.value.fsm_state_seq
                terminals_and_info = last_token.value.terminals_and_info
                final_terminals_and_info = last_token.value.final_terminals_and_info
            else:
                fsm_state_seq = res
                terminals_and_info, final_terminals_and_info = self.scanner.get_terminals_info(fsm_state_seq)
            priority_terminal_info = final_terminals_and_info[0] if final_terminals_and_info else terminals_and_info[0]
            is_not_finished = not priority_terminal_info.is_final or priority_terminal_info.can_transition or len(terminals_and_info) > 1
            start_pos = line_ctr.char_pos
            end_pos = start_pos + len(fsm_state_seq) - 1
            if end_pos >= len(lex_state.text) and is_not_finished:
                type_name = 'partial'
                token_value = PartialTokensInfo(fsm_state_seq, is_not_finished, terminals_and_info, final_terminals_and_info)
                value = ''
            else:
                type_name = priority_terminal_info.terminal_name
                value = token_value = lex_state.text[start_pos:end_pos]
            assert isinstance(self.callback, Dict)
            if type_name not in self.ignore_types:
                t = Token(type_name, token_value, line_ctr.char_pos, line_ctr.line, line_ctr.column)
                line_ctr.feed(value, type_name in self.newline_types)
                t.end_line = line_ctr.line
                t.end_column = line_ctr.column
                t.end_pos = line_ctr.char_pos
                if t.type in self.callback:
                    t = self.callback[t.type](t)
                    if not isinstance(t, Token):
                        raise LexError('Callbacks must return a token (returned %r)' % t)
                lex_state.last_token = t
                return t
            if type_name in self.callback:
                t2 = Token(type_name, value, line_ctr.char_pos, line_ctr.line, line_ctr.column)
                self.callback[type_name](t2)
            line_ctr.feed(value, type_name in self.newline_types)
            last_fsm_state_seq = None
        raise EOFError(self)