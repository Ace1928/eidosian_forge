import re
from pygments.lexer import RegexLexer, bygroups, include, this, using, words
from pygments.token import Comment, Keyword, Literal, Name, Number, \
class AbnfLexer(RegexLexer):
    """
    Lexer for `IETF 7405 ABNF
    <http://www.ietf.org/rfc/rfc7405.txt>`_
    (Updates `5234 <http://www.ietf.org/rfc/rfc5234.txt>`_)
    grammars.

    .. versionadded:: 2.1
    """
    name = 'ABNF'
    aliases = ['abnf']
    filenames = ['*.abnf']
    mimetypes = ['text/x-abnf']
    _core_rules = ('ALPHA', 'BIT', 'CHAR', 'CR', 'CRLF', 'CTL', 'DIGIT', 'DQUOTE', 'HEXDIG', 'HTAB', 'LF', 'LWSP', 'OCTET', 'SP', 'VCHAR', 'WSP')
    tokens = {'root': [(';.*$', Comment.Single), ('(%[si])?"[^"]*"', Literal), ('%b[01]+\\-[01]+\\b', Literal), ('%b[01]+(\\.[01]+)*\\b', Literal), ('%d[0-9]+\\-[0-9]+\\b', Literal), ('%d[0-9]+(\\.[0-9]+)*\\b', Literal), ('%x[0-9a-fA-F]+\\-[0-9a-fA-F]+\\b', Literal), ('%x[0-9a-fA-F]+(\\.[0-9a-fA-F]+)*\\b', Literal), ('\\b[0-9]+\\*[0-9]+', Operator), ('\\b[0-9]+\\*', Operator), ('\\b[0-9]+', Operator), ('\\*', Operator), (words(_core_rules, suffix='\\b'), Keyword), ('[a-zA-Z][a-zA-Z0-9-]+\\b', Name.Class), ('(=/|=|/)', Operator), ('[\\[\\]()]', Punctuation), ('\\s+', Text), ('.', Text)]}