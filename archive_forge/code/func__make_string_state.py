import re
from pygments.lexer import RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def _make_string_state(triple, double, verbatim=None, _escape=_escape):
    if verbatim:
        verbatim = ''.join(['(?:%s|%s)' % (re.escape(c.lower()), re.escape(c.upper())) for c in verbatim])
    char = '"' if double else "'"
    token = String.Double if double else String.Single
    escaped_quotes = '+|%s(?!%s{2})' % (char, char) if triple else ''
    prefix = '%s%s' % ('t' if triple else '', 'd' if double else 's')
    tag_state_name = '%sqt' % prefix
    state = []
    if triple:
        state += [('%s{3,}' % char, token, '#pop'), ('\\\\%s+' % char, String.Escape), (char, token)]
    else:
        state.append((char, token, '#pop'))
    state += [include('s/verbatim'), ('[^\\\\<&{}%s]+' % char, token)]
    if verbatim:
        state.append(('\\\\?<(/|\\\\\\\\|(?!%s)\\\\)%s(?=[\\s=>])' % (_escape, verbatim), Name.Tag, ('#pop', '%sqs' % prefix, tag_state_name)))
    else:
        state += [('\\\\?<!([^><\\\\%s]|<(?!<)|\\\\%s%s|%s|\\\\.)*>?' % (char, char, escaped_quotes, _escape), Comment.Multiline), ('(?i)\\\\?<listing(?=[\\s=>]|\\\\>)', Name.Tag, ('#pop', '%sqs/listing' % prefix, tag_state_name)), ('(?i)\\\\?<xmp(?=[\\s=>]|\\\\>)', Name.Tag, ('#pop', '%sqs/xmp' % prefix, tag_state_name)), ('\\\\?<([^\\s=><\\\\%s]|<(?!<)|\\\\%s%s|%s|\\\\.)*' % (char, char, escaped_quotes, _escape), Name.Tag, tag_state_name), include('s/entity')]
    state += [include('s/escape'), ('\\{([^}<\\\\%s]|<(?!<)|\\\\%s%s|%s|\\\\.)*\\}' % (char, char, escaped_quotes, _escape), String.Interpol), ('[\\\\&{}<]', token)]
    return state