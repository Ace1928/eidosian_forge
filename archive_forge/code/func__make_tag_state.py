import re
from pygments.lexer import RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def _make_tag_state(triple, double, _escape=_escape):
    char = '"' if double else "'"
    quantifier = '{3,}' if triple else ''
    state_name = '%s%sqt' % ('t' if triple else '', 'd' if double else 's')
    token = String.Double if double else String.Single
    escaped_quotes = '+|%s(?!%s{2})' % (char, char) if triple else ''
    return [('%s%s' % (char, quantifier), token, '#pop:2'), ('(\\s|\\\\\\n)+', Text), ('(=)(\\\\?")', bygroups(Punctuation, String.Double), 'dqs/%s' % state_name), ("(=)(\\\\?')", bygroups(Punctuation, String.Single), 'sqs/%s' % state_name), ('=', Punctuation, 'uqs/%s' % state_name), ('\\\\?>', Name.Tag, '#pop'), ('\\{([^}<\\\\%s]|<(?!<)|\\\\%s%s|%s|\\\\.)*\\}' % (char, char, escaped_quotes, _escape), String.Interpol), ('([^\\s=><\\\\%s]|<(?!<)|\\\\%s%s|%s|\\\\.)+' % (char, char, escaped_quotes, _escape), Name.Attribute), include('s/escape'), include('s/verbatim'), include('s/entity'), ('[\\\\{}&]', Name.Attribute)]