import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, bygroups, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def _reset_stringescapes(self):
    self._start = "'"
    self._end = "'"