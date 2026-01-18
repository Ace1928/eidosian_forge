from pygments.lexer import RegexLexer, words, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, \
def _shortened_many(*words):
    return '|'.join(map(_shortened, words))