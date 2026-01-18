import re
from ..core.inputscanner import InputScanner
from ..core.tokenizer import TokenTypes as BaseTokenTypes
from ..core.tokenizer import Tokenizer as BaseTokenizer
from ..core.tokenizer import TokenizerPatterns as BaseTokenizerPatterns
from ..core.directives import Directives
from ..core.pattern import Pattern
from ..core.templatablepattern import TemplatablePattern
def _read_singles(self, c):
    token = None
    if c == '(' or c == '[':
        token = self._create_token(TOKEN.START_EXPR, c)
    elif c == ')' or c == ']':
        token = self._create_token(TOKEN.END_EXPR, c)
    elif c == '{':
        token = self._create_token(TOKEN.START_BLOCK, c)
    elif c == '}':
        token = self._create_token(TOKEN.END_BLOCK, c)
    elif c == ';':
        token = self._create_token(TOKEN.SEMICOLON, c)
    elif c == '.' and self._input.peek(1) is not None and bool(dot_pattern.match(self._input.peek(1))):
        token = self._create_token(TOKEN.DOT, c)
    elif c == ',':
        token = self._create_token(TOKEN.COMMA, c)
    if token is not None:
        self._input.next()
    return token