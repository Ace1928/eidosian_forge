import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _is_list_or_dict_variable(self):
    return self._variable_chars[0] in ('@', '&')