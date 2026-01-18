import re
from pygments.lexer import RegexLexer, include, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class ComponentPascalLexer(RegexLexer):
    """
    For `Component Pascal <http://www.oberon.ch/pdf/CP-Lang.pdf>`_ source code.

    .. versionadded:: 2.1
    """
    name = 'Component Pascal'
    aliases = ['componentpascal', 'cp']
    filenames = ['*.cp', '*.cps']
    mimetypes = ['text/x-component-pascal']
    flags = re.MULTILINE | re.DOTALL
    tokens = {'root': [include('whitespace'), include('comments'), include('punctuation'), include('numliterals'), include('strings'), include('operators'), include('builtins'), include('identifiers')], 'whitespace': [('\\n+', Text), ('\\s+', Text)], 'comments': [('\\(\\*([^$].*?)\\*\\)', Comment.Multiline)], 'punctuation': [('[()\\[\\]{},.:;|]', Punctuation)], 'numliterals': [('[0-9A-F]+X\\b', Number.Hex), ('[0-9A-F]+[HL]\\b', Number.Hex), ('[0-9]+\\.[0-9]+E[+-][0-9]+', Number.Float), ('[0-9]+\\.[0-9]+', Number.Float), ('[0-9]+', Number.Integer)], 'strings': [("'[^\\n']*'", String), ('"[^\\n"]*"', String)], 'operators': [('[+-]', Operator), ('[*/]', Operator), ('[=#<>]', Operator), ('\\^', Operator), ('&', Operator), ('~', Operator), (':=', Operator), ('\\.\\.', Operator), ('\\$', Operator)], 'identifiers': [('([a-zA-Z_$][\\w$]*)', Name)], 'builtins': [(words(('ANYPTR', 'ANYREC', 'BOOLEAN', 'BYTE', 'CHAR', 'INTEGER', 'LONGINT', 'REAL', 'SET', 'SHORTCHAR', 'SHORTINT', 'SHORTREAL'), suffix='\\b'), Keyword.Type), (words(('ABS', 'ABSTRACT', 'ARRAY', 'ASH', 'ASSERT', 'BEGIN', 'BITS', 'BY', 'CAP', 'CASE', 'CHR', 'CLOSE', 'CONST', 'DEC', 'DIV', 'DO', 'ELSE', 'ELSIF', 'EMPTY', 'END', 'ENTIER', 'EXCL', 'EXIT', 'EXTENSIBLE', 'FOR', 'HALT', 'IF', 'IMPORT', 'IN', 'INC', 'INCL', 'IS', 'LEN', 'LIMITED', 'LONG', 'LOOP', 'MAX', 'MIN', 'MOD', 'MODULE', 'NEW', 'ODD', 'OF', 'OR', 'ORD', 'OUT', 'POINTER', 'PROCEDURE', 'RECORD', 'REPEAT', 'RETURN', 'SHORT', 'SHORTCHAR', 'SHORTINT', 'SIZE', 'THEN', 'TYPE', 'TO', 'UNTIL', 'VAR', 'WHILE', 'WITH'), suffix='\\b'), Keyword.Reserved), ('(TRUE|FALSE|NIL|INF)\\b', Keyword.Constant)]}