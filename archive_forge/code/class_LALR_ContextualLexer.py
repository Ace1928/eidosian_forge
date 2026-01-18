import os
from io import open
import types
from functools import wraps, partial
from contextlib import contextmanager
import sys, re
import sre_parse
import sre_constants
from inspect import getmembers, getmro
from functools import partial, wraps
from itertools import repeat, product
class LALR_ContextualLexer(LALR_WithLexer):

    def init_lexer(self):
        states = {idx: list(t.keys()) for idx, t in self.parser._parse_table.states.items()}
        always_accept = self.postlex.always_accept if self.postlex else ()
        self.lexer = ContextualLexer(self.lexer_conf.tokens, states, ignore=self.lexer_conf.ignore, always_accept=always_accept, user_callbacks=self.lexer_conf.callbacks, g_regex_flags=self.lexer_conf.g_regex_flags)

    def parse(self, text, start=None):
        parser_state = [None]

        def set_parser_state(s):
            parser_state[0] = s
        token_stream = self.lex(text, lambda: parser_state[0])
        return self._parse(token_stream, start, set_parser_state)