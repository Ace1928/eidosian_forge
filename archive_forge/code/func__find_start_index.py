import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _find_start_index(self, string, start, end):
    index = string.find('{', start, end) - 1
    if index < 0:
        return -1
    if self._start_index_is_ok(string, index):
        return index
    return self._find_start_index(string, index + 2, end)