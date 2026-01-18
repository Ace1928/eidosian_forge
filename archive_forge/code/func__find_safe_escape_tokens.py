from io import StringIO
from pygments.formatter import Formatter
from pygments.lexer import Lexer, do_insertions
from pygments.token import Token, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt
def _find_safe_escape_tokens(self, text):
    """ find escape tokens that are not in strings or comments """
    for i, t, v in self._filter_to(self.lang.get_tokens_unprocessed(text), lambda t: t in Token.Comment or t in Token.String):
        if t is None:
            for i2, t2, v2 in self._find_escape_tokens(v):
                yield (i + i2, t2, v2)
        else:
            yield (i, None, v)