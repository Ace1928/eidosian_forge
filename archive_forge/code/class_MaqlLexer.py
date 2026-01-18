import re
from pygments.lexer import RegexLexer, include, words, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers._openedge_builtins import OPENEDGEKEYWORDS
class MaqlLexer(RegexLexer):
    """
    Lexer for `GoodData MAQL
    <https://secure.gooddata.com/docs/html/advanced.metric.tutorial.html>`_
    scripts.

    .. versionadded:: 1.4
    """
    name = 'MAQL'
    aliases = ['maql']
    filenames = ['*.maql']
    mimetypes = ['text/x-gooddata-maql', 'application/x-gooddata-maql']
    flags = re.IGNORECASE
    tokens = {'root': [('IDENTIFIER\\b', Name.Builtin), ('\\{[^}]+\\}', Name.Variable), ('[0-9]+(?:\\.[0-9]+)?(?:e[+-]?[0-9]{1,3})?', Number), ('"', String, 'string-literal'), ('\\<\\>|\\!\\=', Operator), ('\\=|\\>\\=|\\>|\\<\\=|\\<', Operator), ('\\:\\=', Operator), ('\\[[^]]+\\]', Name.Variable.Class), (words(('DIMENSION', 'DIMENSIONS', 'BOTTOM', 'METRIC', 'COUNT', 'OTHER', 'FACT', 'WITH', 'TOP', 'OR', 'ATTRIBUTE', 'CREATE', 'PARENT', 'FALSE', 'ROW', 'ROWS', 'FROM', 'ALL', 'AS', 'PF', 'COLUMN', 'COLUMNS', 'DEFINE', 'REPORT', 'LIMIT', 'TABLE', 'LIKE', 'AND', 'BY', 'BETWEEN', 'EXCEPT', 'SELECT', 'MATCH', 'WHERE', 'TRUE', 'FOR', 'IN', 'WITHOUT', 'FILTER', 'ALIAS', 'WHEN', 'NOT', 'ON', 'KEYS', 'KEY', 'FULLSET', 'PRIMARY', 'LABELS', 'LABEL', 'VISUAL', 'TITLE', 'DESCRIPTION', 'FOLDER', 'ALTER', 'DROP', 'ADD', 'DATASET', 'DATATYPE', 'INT', 'BIGINT', 'DOUBLE', 'DATE', 'VARCHAR', 'DECIMAL', 'SYNCHRONIZE', 'TYPE', 'DEFAULT', 'ORDER', 'ASC', 'DESC', 'HYPERLINK', 'INCLUDE', 'TEMPLATE', 'MODIFY'), suffix='\\b'), Keyword), ('[a-z]\\w*\\b', Name.Function), ('#.*', Comment.Single), ('[,;()]', Punctuation), ('\\s+', Text)], 'string-literal': [('\\\\[tnrfbae"\\\\]', String.Escape), ('"', String, '#pop'), ('[^\\\\"]+', String)]}