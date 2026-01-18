import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, words
from pygments.token import Punctuation, Whitespace, Error, \
from pygments.lexers import get_lexer_by_name, ClassNotFound
from pygments.util import iteritems
from pygments.lexers._postgres_builtins import KEYWORDS, DATATYPES, \
from pygments.lexers import _tsql_builtins
class PostgresLexer(PostgresBase, RegexLexer):
    """
    Lexer for the PostgreSQL dialect of SQL.

    .. versionadded:: 1.5
    """
    name = 'PostgreSQL SQL dialect'
    aliases = ['postgresql', 'postgres']
    mimetypes = ['text/x-postgresql']
    flags = re.IGNORECASE
    tokens = {'root': [('\\s+', Text), ('--.*\\n?', Comment.Single), ('/\\*', Comment.Multiline, 'multiline-comments'), ('(' + '|'.join((s.replace(' ', '\\s+') for s in DATATYPES + PSEUDO_TYPES)) + ')\\b', Name.Builtin), (words(KEYWORDS, suffix='\\b'), Keyword), ('[+*/<>=~!@#%^&|`?-]+', Operator), ('::', Operator), ('\\$\\d+', Name.Variable), ('([0-9]*\\.[0-9]*|[0-9]+)(e[+-]?[0-9]+)?', Number.Float), ('[0-9]+', Number.Integer), ("((?:E|U&)?)(')", bygroups(String.Affix, String.Single), 'string'), ('((?:U&)?)(")', bygroups(String.Affix, String.Name), 'quoted-ident'), ('(?s)(\\$)([^$]*)(\\$)(.*?)(\\$)(\\2)(\\$)', language_callback), ('[a-z_]\\w*', Name), (':([\'"]?)[a-z]\\w*\\b\\1', Name.Variable), ('[;:()\\[\\]{},.]', Punctuation)], 'multiline-comments': [('/\\*', Comment.Multiline, 'multiline-comments'), ('\\*/', Comment.Multiline, '#pop'), ('[^/*]+', Comment.Multiline), ('[/*]', Comment.Multiline)], 'string': [("[^']+", String.Single), ("''", String.Single), ("'", String.Single, '#pop')], 'quoted-ident': [('[^"]+', String.Name), ('""', String.Name), ('"', String.Name, '#pop')]}