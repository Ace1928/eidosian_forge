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
class LALR_WithLexer(WithLexer):

    def __init__(self, lexer_conf, parser_conf, options=None):
        debug = options.debug if options else False
        self.parser = LALR_Parser(parser_conf, debug=debug)
        WithLexer.__init__(self, lexer_conf, parser_conf, options)
        self.init_lexer()

    def init_lexer(self):
        raise NotImplementedError()