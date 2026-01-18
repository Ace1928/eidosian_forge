import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, bygroups, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class BroLexer(RegexLexer):
    """
    For `Bro <http://bro-ids.org/>`_ scripts.

    .. versionadded:: 1.5
    """
    name = 'Bro'
    aliases = ['bro']
    filenames = ['*.bro']
    _hex = '[0-9a-fA-F_]'
    _float = '((\\d*\\.?\\d+)|(\\d+\\.?\\d*))([eE][-+]?\\d+)?'
    _h = '[A-Za-z0-9][-A-Za-z0-9]*'
    tokens = {'root': [('^@.*?\\n', Comment.Preproc), ('#.*?\\n', Comment.Single), ('\\n', Text), ('\\s+', Text), ('\\\\\\n', Text), ('(add|alarm|break|case|const|continue|delete|do|else|enum|event|export|for|function|if|global|hook|local|module|next|of|print|redef|return|schedule|switch|type|when|while)\\b', Keyword), ('(addr|any|bool|count|counter|double|file|int|interval|net|pattern|port|record|set|string|subnet|table|time|timer|vector)\\b', Keyword.Type), ('(T|F)\\b', Keyword.Constant), ('(&)((?:add|delete|expire)_func|attr|(?:create|read|write)_expire|default|disable_print_hook|raw_output|encrypt|group|log|mergeable|optional|persistent|priority|redef|rotate_(?:interval|size)|synchronized)\\b', bygroups(Punctuation, Keyword)), ('\\s+module\\b', Keyword.Namespace), ('\\d+/(tcp|udp|icmp|unknown)\\b', Number), ('(\\d+\\.){3}\\d+', Number), ('(' + _hex + '){7}' + _hex, Number), ('0x' + _hex + '(' + _hex + '|:)*::(' + _hex + '|:)*', Number), ('((\\d+|:)(' + _hex + '|:)*)?::(' + _hex + '|:)*', Number), ('(\\d+\\.\\d+\\.|(\\d+\\.){2}\\d+)', Number), (_h + '(\\.' + _h + ')+', String), (_float + '\\s+(day|hr|min|sec|msec|usec)s?\\b', Literal.Date), ('0[xX]' + _hex, Number.Hex), (_float, Number.Float), ('\\d+', Number.Integer), ('/', String.Regex, 'regex'), ('"', String, 'string'), ('[!%*/+:<=>?~|-]', Operator), ('([-+=&|]{2}|[+=!><-]=)', Operator), ('(in|match)\\b', Operator.Word), ('[{}()\\[\\]$.,;]', Punctuation), ('([_a-zA-Z]\\w*)(::)', bygroups(Name, Name.Namespace)), ('[a-zA-Z_]\\w*', Name)], 'string': [('"', String, '#pop'), ('\\\\([\\\\abfnrtv"\\\']|x[a-fA-F0-9]{2,4}|[0-7]{1,3})', String.Escape), ('[^\\\\"\\n]+', String), ('\\\\\\n', String), ('\\\\', String)], 'regex': [('/', String.Regex, '#pop'), ('\\\\[\\\\nt/]', String.Regex), ('[^\\\\/\\n]+', String.Regex), ('\\\\\\n', String.Regex), ('\\\\', String.Regex)]}