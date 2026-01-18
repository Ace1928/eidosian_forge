from io import StringIO
from pygments.formatter import Formatter
from pygments.lexer import Lexer, do_insertions
from pygments.token import Token, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt
def _find_escape_tokens(self, text):
    """ Find escape tokens within text, give token=None otherwise """
    index = 0
    while text:
        a, sep1, text = text.partition(self.left)
        if a:
            yield (index, None, a)
            index += len(a)
        if sep1:
            b, sep2, text = text.partition(self.right)
            if sep2:
                yield (index + len(sep1), Token.Escape, b)
                index += len(sep1) + len(b) + len(sep2)
            else:
                yield (index, Token.Error, sep1)
                index += len(sep1)
                text = b