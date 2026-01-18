import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches
class MSDOSSessionLexer(ShellSessionBaseLexer):
    """
    Lexer for simplistic MSDOS sessions.

    .. versionadded:: 2.1
    """
    name = 'MSDOS Session'
    aliases = ['doscon']
    filenames = []
    mimetypes = []
    _innerLexerCls = BatchLexer
    _ps1rgx = '^([^>]+>)(.*\\n?)'
    _ps2 = 'More? '