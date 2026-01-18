import numba.core.config
from pygments.styles.manni import ManniStyle
from pygments.styles.monokai import MonokaiStyle
from pygments.styles.native import NativeStyle
from pygments.lexer import RegexLexer, include, bygroups, words
from pygments.token import Text, Name, String,  Punctuation, Keyword, \
from pygments.style import Style
class NumbaIRLexer(RegexLexer):
    """
    Pygments style lexer for Numba IR (for use with highlighting etc).
    """
    name = 'Numba_IR'
    aliases = ['numba_ir']
    filenames = ['*.numba_ir']
    identifier = '\\$[a-zA-Z0-9._]+'
    fun_or_var = '([a-zA-Z_]+[a-zA-Z0-9]*)'
    tokens = {'root': [('(label)(\\ [0-9]+)(:)$', bygroups(Keyword, Name.Label, Punctuation)), (' = ', Operator), include('whitespace'), include('keyword'), (identifier, Name.Variable), (fun_or_var + '(\\()', bygroups(Name.Function, Punctuation)), (fun_or_var + '(\\=)', bygroups(Name.Attribute, Punctuation)), (fun_or_var, Name.Constant), ('[0-9]+', Number), ('<[^>\\n]*>', String), ("[=<>{}\\[\\]()*.,!\\':]|x\\b", Punctuation)], 'keyword': [(words(('del', 'jump', 'call', 'branch'), suffix=' '), Keyword)], 'whitespace': [('(\\n|\\s)', Text)]}