import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class ForLoop(Tokenizer):

    def __init__(self):
        Tokenizer.__init__(self)
        self._in_arguments = False

    def _tokenize(self, value, index):
        token = self._in_arguments and ARGUMENT or SYNTAX
        if value.upper() in ('IN', 'IN RANGE'):
            self._in_arguments = True
        return token