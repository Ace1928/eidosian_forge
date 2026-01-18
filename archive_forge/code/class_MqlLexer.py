import re
from pygments.lexer import RegexLexer, include, bygroups, inherit, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers import _mql_builtins
class MqlLexer(CppLexer):
    """
    For `MQL4 <http://docs.mql4.com/>`_ and
    `MQL5 <http://www.mql5.com/en/docs>`_ source code.

    .. versionadded:: 2.0
    """
    name = 'MQL'
    aliases = ['mql', 'mq4', 'mq5', 'mql4', 'mql5']
    filenames = ['*.mq4', '*.mq5', '*.mqh']
    mimetypes = ['text/x-mql']
    tokens = {'statements': [(words(_mql_builtins.keywords, suffix='\\b'), Keyword), (words(_mql_builtins.c_types, suffix='\\b'), Keyword.Type), (words(_mql_builtins.types, suffix='\\b'), Name.Function), (words(_mql_builtins.constants, suffix='\\b'), Name.Constant), (words(_mql_builtins.colors, prefix='(clr)?', suffix='\\b'), Name.Constant), inherit]}