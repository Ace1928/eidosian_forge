import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, words
from pygments.token import Punctuation, Whitespace, Error, \
from pygments.lexers import get_lexer_by_name, ClassNotFound
from pygments.util import iteritems
from pygments.lexers._postgres_builtins import KEYWORDS, DATATYPES, \
from pygments.lexers import _tsql_builtins
class lookahead(object):
    """Wrap an iterator and allow pushing back an item."""

    def __init__(self, x):
        self.iter = iter(x)
        self._nextitem = None

    def __iter__(self):
        return self

    def send(self, i):
        self._nextitem = i
        return i

    def __next__(self):
        if self._nextitem is not None:
            ni = self._nextitem
            self._nextitem = None
            return ni
        return next(self.iter)
    next = __next__