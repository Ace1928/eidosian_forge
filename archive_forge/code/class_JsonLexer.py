import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, LexerContext, \
from pygments.token import Text, Comment, Keyword, Name, String, Number, \
class JsonLexer(RegexLexer):
    """
    For JSON data structures.

    .. versionadded:: 1.5
    """
    name = 'JSON'
    aliases = ['json']
    filenames = ['*.json']
    mimetypes = ['application/json']
    flags = re.DOTALL
    int_part = '-?(0|[1-9]\\d*)'
    frac_part = '\\.\\d+'
    exp_part = '[eE](\\+|-)?\\d+'
    tokens = {'whitespace': [('\\s+', Text)], 'simplevalue': [('(true|false|null)\\b', Keyword.Constant), ('%(int_part)s(%(frac_part)s%(exp_part)s|%(exp_part)s|%(frac_part)s)' % vars(), Number.Float), (int_part, Number.Integer), ('"(\\\\\\\\|\\\\"|[^"])*"', String.Double)], 'objectattribute': [include('value'), (':', Punctuation), (',', Punctuation, '#pop'), ('\\}', Punctuation, '#pop:2')], 'objectvalue': [include('whitespace'), ('"(\\\\\\\\|\\\\"|[^"])*"', Name.Tag, 'objectattribute'), ('\\}', Punctuation, '#pop')], 'arrayvalue': [include('whitespace'), include('value'), (',', Punctuation), ('\\]', Punctuation, '#pop')], 'value': [include('whitespace'), include('simplevalue'), ('\\{', Punctuation, 'objectvalue'), ('\\[', Punctuation, 'arrayvalue')], 'root': [include('value')]}