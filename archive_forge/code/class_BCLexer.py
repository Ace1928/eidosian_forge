import re
from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class BCLexer(RegexLexer):
    """
    A `BC <https://www.gnu.org/software/bc/>`_ lexer.

    .. versionadded:: 2.1
    """
    name = 'BC'
    aliases = ['bc']
    filenames = ['*.bc']
    tokens = {'root': [('/\\*', Comment.Multiline, 'comment'), ('"(?:[^"\\\\]|\\\\.)*"', String), ('[{}();,]', Punctuation), (words(('if', 'else', 'while', 'for', 'break', 'continue', 'halt', 'return', 'define', 'auto', 'print', 'read', 'length', 'scale', 'sqrt', 'limits', 'quit', 'warranty'), suffix='\\b'), Keyword), ('\\+\\+|--|\\|\\||&&|([-<>+*%\\^/!=])=?', Operator), ('[0-9]+(\\.[0-9]*)?', Number), ('\\.[0-9]+', Number), ('.', Text)], 'comment': [('[^*/]+', Comment.Multiline), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)]}