import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches
class PowerShellSessionLexer(ShellSessionBaseLexer):
    """
    Lexer for simplistic Windows PowerShell sessions.

    .. versionadded:: 2.1
    """
    name = 'PowerShell Session'
    aliases = ['ps1con']
    filenames = []
    mimetypes = []
    _innerLexerCls = PowerShellLexer
    _ps1rgx = '^(PS [^>]+> )(.*\\n?)'
    _ps2 = '>> '