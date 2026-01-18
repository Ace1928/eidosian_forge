import sys
import __future__
import inspect
import tokenize
import ast
import numbers
import six
from patsy import PatsyError
from patsy.util import PushbackAdapter, no_pickling, assert_no_pickling
from patsy.tokens import (pretty_untokenize, normalize_token_spacing,
from patsy.compat import call_and_wrap_exc
import patsy.builtins
class _FuncallCapturer(object):

    def __init__(self, start_token_type, start_token):
        self.func = [start_token]
        self.tokens = [(start_token_type, start_token)]
        self.paren_depth = 0
        self.started = False
        self.done = False

    def add_token(self, token_type, token):
        if self.done:
            return
        self.tokens.append((token_type, token))
        if token in ['(', '{', '[']:
            self.paren_depth += 1
        if token in [')', '}', ']']:
            self.paren_depth -= 1
        assert self.paren_depth >= 0
        if not self.started:
            if token == '(':
                self.started = True
            else:
                assert token_type == tokenize.NAME or token == '.'
                self.func.append(token)
        if self.started and self.paren_depth == 0:
            self.done = True