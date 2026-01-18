import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches
class TcshSessionLexer(ShellSessionBaseLexer):
    """
    Lexer for Tcsh sessions.

    .. versionadded:: 2.1
    """
    name = 'Tcsh Session'
    aliases = ['tcshcon']
    filenames = []
    mimetypes = []
    _innerLexerCls = TcshLexer
    _ps1rgx = '^([^>]+>)(.*\\n?)'
    _ps2 = '? '