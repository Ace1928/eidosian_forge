import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, bygroups, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def _stringescapes(lexer, match, ctx):
    lexer._start = match.group(3)
    lexer._end = match.group(5)
    return bygroups(Keyword.Reserved, Text, String.Escape, Text, String.Escape)(lexer, match, ctx)