import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class TerminfoLexer(RegexLexer):
    """
    Lexer for terminfo database source.

    This is very simple and minimal.

    .. versionadded:: 2.1
    """
    name = 'Terminfo'
    aliases = ['terminfo']
    filenames = ['terminfo', 'terminfo.src']
    mimetypes = []
    tokens = {'root': [('^#.*$', Comment), ('^[^\\s#,|]+', Name.Tag, 'names')], 'names': [('\\n', Text, '#pop'), ('(,)([ \\t]*)', bygroups(Punctuation, Text), 'defs'), ('\\|', Punctuation), ('[^,|]+', Name.Attribute)], 'defs': [('\\n[ \\t]+', Text), ('\\n', Text, '#pop:2'), ('(#)([0-9]+)', bygroups(Operator, Number)), ('=', Operator, 'data'), ('(,)([ \\t]*)', bygroups(Punctuation, Text)), ('[^\\s,=#]+', Name.Class)], 'data': [('\\\\[,\\\\]', Literal), ('(,)([ \\t]*)', bygroups(Punctuation, Text), '#pop'), ('[^\\\\,]+', Literal), ('.', Literal)]}