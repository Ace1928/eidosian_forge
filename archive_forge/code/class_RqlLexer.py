import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, words
from pygments.token import Punctuation, Whitespace, Error, \
from pygments.lexers import get_lexer_by_name, ClassNotFound
from pygments.util import iteritems
from pygments.lexers._postgres_builtins import KEYWORDS, DATATYPES, \
from pygments.lexers import _tsql_builtins
class RqlLexer(RegexLexer):
    """
    Lexer for Relation Query Language.

    `RQL <http://www.logilab.org/project/rql>`_

    .. versionadded:: 2.0
    """
    name = 'RQL'
    aliases = ['rql']
    filenames = ['*.rql']
    mimetypes = ['text/x-rql']
    flags = re.IGNORECASE
    tokens = {'root': [('\\s+', Text), ('(DELETE|SET|INSERT|UNION|DISTINCT|WITH|WHERE|BEING|OR|AND|NOT|GROUPBY|HAVING|ORDERBY|ASC|DESC|LIMIT|OFFSET|TODAY|NOW|TRUE|FALSE|NULL|EXISTS)\\b', Keyword), ('[+*/<>=%-]', Operator), ('(Any|is|instance_of|CWEType|CWRelation)\\b', Name.Builtin), ('[0-9]+', Number.Integer), ('[A-Z_]\\w*\\??', Name), ("'(''|[^'])*'", String.Single), ('"(""|[^"])*"', String.Single), ('[;:()\\[\\],.]', Punctuation)]}