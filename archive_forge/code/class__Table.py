import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class _Table(object):
    _tokenizer_class = None

    def __init__(self, prev_tokenizer=None):
        self._tokenizer = self._tokenizer_class()
        self._prev_tokenizer = prev_tokenizer
        self._prev_values_on_row = []

    def tokenize(self, value, index):
        if self._continues(value, index):
            self._tokenizer = self._prev_tokenizer
            yield (value, SYNTAX)
        else:
            for value_and_token in self._tokenize(value, index):
                yield value_and_token
        self._prev_values_on_row.append(value)

    def _continues(self, value, index):
        return value == '...' and all((self._is_empty(t) for t in self._prev_values_on_row))

    def _is_empty(self, value):
        return value in ('', '\\')

    def _tokenize(self, value, index):
        return self._tokenizer.tokenize(value)

    def end_row(self):
        self.__init__(prev_tokenizer=self._tokenizer)