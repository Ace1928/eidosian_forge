import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, words
from pygments.token import Punctuation, Whitespace, Error, \
from pygments.lexers import get_lexer_by_name, ClassNotFound
from pygments.util import iteritems
from pygments.lexers._postgres_builtins import KEYWORDS, DATATYPES, \
from pygments.lexers import _tsql_builtins
class TransactSqlLexer(RegexLexer):
    """
    Transact-SQL (T-SQL) is Microsoft's and Sybase's proprietary extension to
    SQL.

    The list of keywords includes ODBC and keywords reserved for future use..
    """
    name = 'Transact-SQL'
    aliases = ['tsql', 't-sql']
    filenames = ['*.sql']
    mimetypes = ['text/x-tsql']
    flags = re.IGNORECASE | re.UNICODE
    tokens = {'root': [('\\s+', Whitespace), ('--(?m).*?$\\n?', Comment.Single), ('/\\*', Comment.Multiline, 'multiline-comments'), (words(_tsql_builtins.OPERATORS), Operator), (words(_tsql_builtins.OPERATOR_WORDS, suffix='\\b'), Operator.Word), (words(_tsql_builtins.TYPES, suffix='\\b'), Name.Class), (words(_tsql_builtins.FUNCTIONS, suffix='\\b'), Name.Function), ('(goto)(\\s+)(\\w+\\b)', bygroups(Keyword, Whitespace, Name.Label)), (words(_tsql_builtins.KEYWORDS, suffix='\\b'), Keyword), ('(\\[)([^]]+)(\\])', bygroups(Operator, Name, Operator)), ('0x[0-9a-f]+', Number.Hex), ('[0-9]+\\.[0-9]*(e[+-]?[0-9]+)?', Number.Float), ('\\.[0-9]+(e[+-]?[0-9]+)?', Number.Float), ('[0-9]+e[+-]?[0-9]+', Number.Float), ('[0-9]+', Number.Integer), ("'(''|[^'])*'", String.Single), ('"(""|[^"])*"', String.Symbol), ('[;(),.]', Punctuation), ('@@\\w+', Name.Builtin), ('@\\w+', Name.Variable), ('(\\w+)(:)', bygroups(Name.Label, Punctuation)), ('#?#?\\w+', Name), ('\\?', Name.Variable.Magic)], 'multiline-comments': [('/\\*', Comment.Multiline, 'multiline-comments'), ('\\*/', Comment.Multiline, '#pop'), ('[^/*]+', Comment.Multiline), ('[/*]', Comment.Multiline)]}