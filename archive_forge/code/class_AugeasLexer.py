import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, default, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
from pygments.lexers.data import JsonLexer
class AugeasLexer(RegexLexer):
    """
    Lexer for Augeas.

    .. versionadded:: 2.4
    """
    name = 'Augeas'
    url = 'http://augeas.net'
    aliases = ['augeas']
    filenames = ['*.aug']
    tokens = {'root': [('(module)(\\s*)([^\\s=]+)', bygroups(Keyword.Namespace, Whitespace, Name.Namespace)), ('(let)(\\s*)([^\\s=]+)', bygroups(Keyword.Declaration, Whitespace, Name.Variable)), ('(del|store|value|counter|seq|key|label|autoload|incl|excl|transform|test|get|put)(\\s+)', bygroups(Name.Builtin, Whitespace)), ('(\\()([^:]+)(\\:)(unit|string|regexp|lens|tree|filter)(\\))', bygroups(Punctuation, Name.Variable, Punctuation, Keyword.Type, Punctuation)), ('\\(\\*', Comment.Multiline, 'comment'), ('[*+\\-.;=?|]', Operator), ('[()\\[\\]{}]', Operator), ('"', String.Double, 'string'), ('\\/', String.Regex, 'regex'), ('([A-Z]\\w*)(\\.)(\\w+)', bygroups(Name.Namespace, Punctuation, Name.Variable)), ('.', Name.Variable), ('\\s+', Whitespace)], 'string': [('\\\\.', String.Escape), ('[^"]', String.Double), ('"', String.Double, '#pop')], 'regex': [('\\\\.', String.Escape), ('[^/]', String.Regex), ('\\/', String.Regex, '#pop')], 'comment': [('[^*)]', Comment.Multiline), ('\\(\\*', Comment.Multiline, '#push'), ('\\*\\)', Comment.Multiline, '#pop'), ('[)*]', Comment.Multiline)]}