import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _is_escaped(self, string, index):
    escaped = False
    while index > 0 and string[index - 1] == '\\':
        index -= 1
        escaped = not escaped
    return escaped