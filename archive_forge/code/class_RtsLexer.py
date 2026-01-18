import re
from pygments.lexer import RegexLexer
from pygments.token import String, Number, Name, Keyword, Operator, Text, Comment
class RtsLexer(RegexLexer):
    """
    For `Riverbed Stingray Traffic Manager <http://www.riverbed.com/stingray>`_

    .. versionadded:: 2.1
    """
    name = 'TrafficScript'
    aliases = ['rts', 'trafficscript']
    filenames = ['*.rts']
    tokens = {'root': [("'(\\\\\\\\|\\\\[^\\\\]|[^'\\\\])*'", String), ('"', String, 'escapable-string'), ('(0x[0-9a-fA-F]+|\\d+)', Number), ('\\d+\\.\\d+', Number.Float), ('\\$[a-zA-Z](\\w|_)*', Name.Variable), ('(if|else|for(each)?|in|while|do|break|sub|return|import)', Keyword), ('[a-zA-Z][\\w.]*', Name.Function), ('[-+*/%=,;(){}<>^.!~|&\\[\\]\\?\\:]', Operator), ('(>=|<=|==|!=|&&|\\|\\||\\+=|.=|-=|\\*=|/=|%=|<<=|>>=|&=|\\|=|\\^=|>>|<<|\\+\\+|--|=>)', Operator), ('[ \\t\\r]+', Text), ('#[^\\n]*', Comment)], 'escapable-string': [('\\\\[tsn]', String.Escape), ('[^"]', String), ('"', String, '#pop')]}