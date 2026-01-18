import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _continues(self, value, index):
    return index > 0 and _Table._continues(self, value, index)