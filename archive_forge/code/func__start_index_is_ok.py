import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _start_index_is_ok(self, string, index):
    return string[index] in self._identifiers and (not self._is_escaped(string, index))