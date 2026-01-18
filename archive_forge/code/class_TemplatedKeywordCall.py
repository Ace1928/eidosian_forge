import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class TemplatedKeywordCall(Tokenizer):
    _tokens = (ARGUMENT,)