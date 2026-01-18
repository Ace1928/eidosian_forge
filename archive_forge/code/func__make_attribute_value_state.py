import re
from pygments.lexer import RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def _make_attribute_value_state(terminator, host_triple, host_double, _escape=_escape):
    token = String.Double if terminator == '"' else String.Single if terminator == "'" else String.Other
    host_char = '"' if host_double else "'"
    host_quantifier = '{3,}' if host_triple else ''
    host_token = String.Double if host_double else String.Single
    escaped_quotes = '+|%s(?!%s{2})' % (host_char, host_char) if host_triple else ''
    return [('%s%s' % (host_char, host_quantifier), host_token, '#pop:3'), ('%s%s' % ('' if token is String.Other else '\\\\?', terminator), token, '#pop'), include('s/verbatim'), include('s/entity'), ('\\{([^}<\\\\%s]|<(?!<)|\\\\%s%s|%s|\\\\.)*\\}' % (host_char, host_char, escaped_quotes, _escape), String.Interpol), ('([^\\s"\\\'<%s{}\\\\&])+' % ('>' if token is String.Other else ''), token), include('s/escape'), ('["\\\'\\s&{<}\\\\]', token)]