import re
from pygments.lexer import RegexLexer, include, words, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers._openedge_builtins import OPENEDGEKEYWORDS
class OpenEdgeLexer(RegexLexer):
    """
    Lexer for `OpenEdge ABL (formerly Progress)
    <http://web.progress.com/en/openedge/abl.html>`_ source code.

    .. versionadded:: 1.5
    """
    name = 'OpenEdge ABL'
    aliases = ['openedge', 'abl', 'progress']
    filenames = ['*.p', '*.cls']
    mimetypes = ['text/x-openedge', 'application/x-openedge']
    types = '(?i)(^|(?<=[^\\w\\-]))(CHARACTER|CHAR|CHARA|CHARAC|CHARACT|CHARACTE|COM-HANDLE|DATE|DATETIME|DATETIME-TZ|DECIMAL|DEC|DECI|DECIM|DECIMA|HANDLE|INT64|INTEGER|INT|INTE|INTEG|INTEGE|LOGICAL|LONGCHAR|MEMPTR|RAW|RECID|ROWID)\\s*($|(?=[^\\w\\-]))'
    keywords = words(OPENEDGEKEYWORDS, prefix='(?i)(^|(?<=[^\\w\\-]))', suffix='\\s*($|(?=[^\\w\\-]))')
    tokens = {'root': [('/\\*', Comment.Multiline, 'comment'), ('\\{', Comment.Preproc, 'preprocessor'), ('\\s*&.*', Comment.Preproc), ('0[xX][0-9a-fA-F]+[LlUu]*', Number.Hex), ('(?i)(DEFINE|DEF|DEFI|DEFIN)\\b', Keyword.Declaration), (types, Keyword.Type), (keywords, Name.Builtin), ('"(\\\\\\\\|\\\\"|[^"])*"', String.Double), ("'(\\\\\\\\|\\\\'|[^'])*'", String.Single), ('[0-9][0-9]*\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('[0-9]+', Number.Integer), ('\\s+', Text), ('[+*/=-]', Operator), ('[.:()]', Punctuation), ('.', Name.Variable)], 'comment': [('[^*/]', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)], 'preprocessor': [('[^{}]', Comment.Preproc), ('\\{', Comment.Preproc, '#push'), ('\\}', Comment.Preproc, '#pop')]}