import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class KeywordCall(Tokenizer):
    _tokens = (KEYWORD, ARGUMENT)

    def __init__(self, support_assign=True):
        Tokenizer.__init__(self)
        self._keyword_found = not support_assign
        self._assigns = 0

    def _tokenize(self, value, index):
        if not self._keyword_found and self._is_assign(value):
            self._assigns += 1
            return SYNTAX
        if self._keyword_found:
            return Tokenizer._tokenize(self, value, index - self._assigns)
        self._keyword_found = True
        return GherkinTokenizer().tokenize(value, KEYWORD)