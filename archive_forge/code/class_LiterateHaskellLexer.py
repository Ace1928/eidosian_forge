import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class LiterateHaskellLexer(LiterateLexer):
    """
    For Literate Haskell (Bird-style or LaTeX) source.

    Additional options accepted:

    `litstyle`
        If given, must be ``"bird"`` or ``"latex"``.  If not given, the style
        is autodetected: if the first non-whitespace character in the source
        is a backslash or percent character, LaTeX is assumed, else Bird.

    .. versionadded:: 0.9
    """
    name = 'Literate Haskell'
    aliases = ['lhs', 'literate-haskell', 'lhaskell']
    filenames = ['*.lhs']
    mimetypes = ['text/x-literate-haskell']

    def __init__(self, **options):
        hslexer = HaskellLexer(**options)
        LiterateLexer.__init__(self, hslexer, **options)