import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _scanning_list_variable_index(self):
    return self._state in [self._waiting_list_variable_index_state, self._list_variable_index_state]