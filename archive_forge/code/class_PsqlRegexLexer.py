import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, words
from pygments.token import Punctuation, Whitespace, Error, \
from pygments.lexers import get_lexer_by_name, ClassNotFound
from pygments.util import iteritems
from pygments.lexers._postgres_builtins import KEYWORDS, DATATYPES, \
from pygments.lexers import _tsql_builtins
class PsqlRegexLexer(PostgresBase, RegexLexer):
    """
    Extend the PostgresLexer adding support specific for psql commands.

    This is not a complete psql lexer yet as it lacks prompt support
    and output rendering.
    """
    name = 'PostgreSQL console - regexp based lexer'
    aliases = []
    flags = re.IGNORECASE
    tokens = dict(((k, l[:]) for k, l in iteritems(PostgresLexer.tokens)))
    tokens['root'].append(('\\\\[^\\s]+', Keyword.Pseudo, 'psql-command'))
    tokens['psql-command'] = [('\\n', Text, 'root'), ('\\s+', Text), ('\\\\[^\\s]+', Keyword.Pseudo), (':([\'"]?)[a-z]\\w*\\b\\1', Name.Variable), ("'(''|[^'])*'", String.Single), ('`([^`])*`', String.Backtick), ('[^\\s]+', String.Symbol)]