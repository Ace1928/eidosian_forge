import re
from pygments.lexer import RegexLexer, include, words, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers._openedge_builtins import OPENEDGEKEYWORDS
class CobolFreeformatLexer(CobolLexer):
    """
    Lexer for Free format OpenCOBOL code.

    .. versionadded:: 1.6
    """
    name = 'COBOLFree'
    aliases = ['cobolfree']
    filenames = ['*.cbl', '*.CBL']
    mimetypes = []
    flags = re.IGNORECASE | re.MULTILINE
    tokens = {'comment': [('(\\*>.*\\n|^\\w*\\*.*$)', Comment)]}