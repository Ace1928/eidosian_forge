import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _start_table(self, header):
    name = normalize(header, remove='*')
    return self._tables.get(name, UnknownTable())