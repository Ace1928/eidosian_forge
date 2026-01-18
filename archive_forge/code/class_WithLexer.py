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
class WithLexer(_ParserFrontend):
    lexer = None
    parser = None
    lexer_conf = None
    start = None
    __serialize_fields__ = ('parser', 'lexer_conf', 'start')
    __serialize_namespace__ = (LexerConf,)

    def __init__(self, lexer_conf, parser_conf, options=None):
        self.lexer_conf = lexer_conf
        self.start = parser_conf.start
        self.postlex = lexer_conf.postlex

    @classmethod
    def deserialize(cls, data, memo, callbacks, postlex):
        inst = super(WithLexer, cls).deserialize(data, memo)
        inst.postlex = postlex
        inst.parser = LALR_Parser.deserialize(inst.parser, memo, callbacks)
        inst.init_lexer()
        return inst

    def _serialize(self, data, memo):
        data['parser'] = data['parser'].serialize(memo)

    def lex(self, *args):
        stream = self.lexer.lex(*args)
        return self.postlex.process(stream) if self.postlex else stream

    def parse(self, text, start=None):
        token_stream = self.lex(text)
        return self._parse(token_stream, start)

    def init_traditional_lexer(self):
        self.lexer = TraditionalLexer(self.lexer_conf.tokens, ignore=self.lexer_conf.ignore, user_callbacks=self.lexer_conf.callbacks, g_regex_flags=self.lexer_conf.g_regex_flags)