from pygments.lexer import RegexLexer, words, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, \
def _shortened(word):
    dpos = word.find('$')
    return '|'.join((word[:dpos] + word[dpos + 1:i] + '\\b' for i in range(len(word), dpos, -1)))