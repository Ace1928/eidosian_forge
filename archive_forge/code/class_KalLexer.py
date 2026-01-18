import re
from pygments.lexer import RegexLexer, include, bygroups, default, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, iteritems
import pygments.unistring as uni
class KalLexer(RegexLexer):
    """
    For `Kal`_ source code.

    .. _Kal: http://rzimmerman.github.io/kal


    .. versionadded:: 2.0
    """
    name = 'Kal'
    aliases = ['kal']
    filenames = ['*.kal']
    mimetypes = ['text/kal', 'application/kal']
    flags = re.DOTALL
    tokens = {'commentsandwhitespace': [('\\s+', Text), ('###[^#].*?###', Comment.Multiline), ('#(?!##[^#]).*?\\n', Comment.Single)], 'functiondef': [('[$a-zA-Z_][\\w$]*\\s*', Name.Function, '#pop'), include('commentsandwhitespace')], 'classdef': [('\\binherits\\s+from\\b', Keyword), ('[$a-zA-Z_][\\w$]*\\s*\\n', Name.Class, '#pop'), ('[$a-zA-Z_][\\w$]*\\s*', Name.Class), include('commentsandwhitespace')], 'listcomprehension': [('\\]', Punctuation, '#pop'), ('\\b(property|value)\\b', Keyword), include('root')], 'waitfor': [('\\n', Punctuation, '#pop'), ('\\bfrom\\b', Keyword), include('root')], 'root': [include('commentsandwhitespace'), ('/(?! )(\\\\.|[^[/\\\\\\n]|\\[(\\\\.|[^\\]\\\\\\n])*])+/([gim]+\\b|\\B)', String.Regex), ('\\?|:|_(?=\\n)|==?|!=|-(?!>)|[<>+*/-]=?', Operator), ('\\b(and|or|isnt|is|not|but|bitwise|mod|\\^|xor|exists|doesnt\\s+exist)\\b', Operator.Word), ('(?:\\([^()]+\\))?\\s*>', Name.Function), ('[{(]', Punctuation), ('\\[', Punctuation, 'listcomprehension'), ('[})\\].,]', Punctuation), ('\\b(function|method|task)\\b', Keyword.Declaration, 'functiondef'), ('\\bclass\\b', Keyword.Declaration, 'classdef'), ('\\b(safe\\s+)?wait\\s+for\\b', Keyword, 'waitfor'), ('\\b(me|this)(\\.[$a-zA-Z_][\\w.$]*)?\\b', Name.Variable.Instance), ('(?<![.$])(for(\\s+(parallel|series))?|in|of|while|until|break|return|continue|when|if|unless|else|otherwise|except\\s+when|throw|raise|fail\\s+with|try|catch|finally|new|delete|typeof|instanceof|super|run\\s+in\\s+parallel|inherits\\s+from)\\b', Keyword), ('(?<![.$])(true|false|yes|no|on|off|null|nothing|none|NaN|Infinity|undefined)\\b', Keyword.Constant), ('(Array|Boolean|Date|Error|Function|Math|netscape|Number|Object|Packages|RegExp|String|sun|decodeURI|decodeURIComponent|encodeURI|encodeURIComponent|eval|isFinite|isNaN|isSafeInteger|parseFloat|parseInt|document|window|print)\\b', Name.Builtin), ('[$a-zA-Z_][\\w.$]*\\s*(:|[+\\-*/]?\\=)?\\b', Name.Variable), ('[0-9][0-9]*\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('0x[0-9a-fA-F]+', Number.Hex), ('[0-9]+', Number.Integer), ('"""', String, 'tdqs'), ("'''", String, 'tsqs'), ('"', String, 'dqs'), ("'", String, 'sqs')], 'strings': [('[^#\\\\\\\'"]+', String)], 'interpoling_string': [('\\}', String.Interpol, '#pop'), include('root')], 'dqs': [('"', String, '#pop'), ("\\\\.|\\'", String), ('#\\{', String.Interpol, 'interpoling_string'), include('strings')], 'sqs': [("'", String, '#pop'), ('#|\\\\.|"', String), include('strings')], 'tdqs': [('"""', String, '#pop'), ('\\\\.|\\\'|"', String), ('#\\{', String.Interpol, 'interpoling_string'), include('strings')], 'tsqs': [("'''", String, '#pop'), ('#|\\\\.|\\\'|"', String), include('strings')]}