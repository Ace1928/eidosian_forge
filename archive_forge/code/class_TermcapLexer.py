import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class TermcapLexer(RegexLexer):
    """
    Lexer for termcap database source.

    This is very simple and minimal.

    .. versionadded:: 2.1
    """
    name = 'Termcap'
    aliases = ['termcap']
    filenames = ['termcap', 'termcap.src']
    mimetypes = []
    tokens = {'root': [('^#.*$', Comment), ('^[^\\s#:|]+', Name.Tag, 'names')], 'names': [('\\n', Text, '#pop'), (':', Punctuation, 'defs'), ('\\|', Punctuation), ('[^:|]+', Name.Attribute)], 'defs': [('\\\\\\n[ \\t]*', Text), ('\\n[ \\t]*', Text, '#pop:2'), ('(#)([0-9]+)', bygroups(Operator, Number)), ('=', Operator, 'data'), (':', Punctuation), ('[^\\s:=#]+', Name.Class)], 'data': [('\\\\072', Literal), (':', Punctuation, '#pop'), ('[^:\\\\]+', Literal), ('.', Literal)]}