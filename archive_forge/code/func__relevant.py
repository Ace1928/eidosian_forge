import re
from pygments.lexer import RegexLexer, include, bygroups, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
def _relevant(self, token):
    return token not in (Text, Comment.Single, Comment.Multiline)