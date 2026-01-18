import numba.core.config
from pygments.styles.manni import ManniStyle
from pygments.styles.monokai import MonokaiStyle
from pygments.styles.native import NativeStyle
from pygments.lexer import RegexLexer, include, bygroups, words
from pygments.token import Text, Name, String,  Punctuation, Keyword, \
from pygments.style import Style

    Get appropriate style for highlighting according to
    NUMBA_COLOR_SCHEME setting
    