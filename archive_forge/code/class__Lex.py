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
class _Lex:
    """Built to serve both Lexer and ContextualLexer"""

    def __init__(self, lexer, state=None):
        self.lexer = lexer
        self.state = state

    def lex(self, stream, newline_types, ignore_types):
        newline_types = frozenset(newline_types)
        ignore_types = frozenset(ignore_types)
        line_ctr = LineCounter()
        last_token = None
        while line_ctr.char_pos < len(stream):
            lexer = self.lexer
            res = lexer.match(stream, line_ctr.char_pos)
            if not res:
                allowed = {v for m, tfi in lexer.mres for v in tfi.values()} - ignore_types
                if not allowed:
                    allowed = {'<END-OF-FILE>'}
                raise UnexpectedCharacters(stream, line_ctr.char_pos, line_ctr.line, line_ctr.column, allowed=allowed, state=self.state, token_history=last_token and [last_token])
            value, type_ = res
            if type_ not in ignore_types:
                t = Token(type_, value, line_ctr.char_pos, line_ctr.line, line_ctr.column)
                line_ctr.feed(value, type_ in newline_types)
                t.end_line = line_ctr.line
                t.end_column = line_ctr.column
                t.end_pos = line_ctr.char_pos
                if t.type in lexer.callback:
                    t = lexer.callback[t.type](t)
                    if not isinstance(t, Token):
                        raise ValueError('Callbacks must return a token (returned %r)' % t)
                yield t
                last_token = t
            else:
                if type_ in lexer.callback:
                    t2 = Token(type_, value, line_ctr.char_pos, line_ctr.line, line_ctr.column)
                    lexer.callback[type_](t2)
                line_ctr.feed(value, type_ in newline_types)