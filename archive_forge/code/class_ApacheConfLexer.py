import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class ApacheConfLexer(RegexLexer):
    """
    Lexer for configuration files following the Apache config file
    format.

    .. versionadded:: 0.6
    """
    name = 'ApacheConf'
    aliases = ['apacheconf', 'aconf', 'apache']
    filenames = ['.htaccess', 'apache.conf', 'apache2.conf']
    mimetypes = ['text/x-apacheconf']
    flags = re.MULTILINE | re.IGNORECASE
    tokens = {'root': [('\\s+', Text), ('(#.*?)$', Comment), ('(<[^\\s>]+)(?:(\\s+)(.*?))?(>)', bygroups(Name.Tag, Text, String, Name.Tag)), ('([a-z]\\w*)(\\s+)', bygroups(Name.Builtin, Text), 'value'), ('\\.+', Text)], 'value': [('\\\\\\n', Text), ('$', Text, '#pop'), ('\\\\', Text), ('[^\\S\\n]+', Text), ('\\d+\\.\\d+\\.\\d+\\.\\d+(?:/\\d+)?', Number), ('\\d+', Number), ('/([a-z0-9][\\w./-]+)', String.Other), ('(on|off|none|any|all|double|email|dns|min|minimal|os|productonly|full|emerg|alert|crit|error|warn|notice|info|debug|registry|script|inetd|standalone|user|group)\\b', Keyword), ('"([^"\\\\]*(?:\\\\.[^"\\\\]*)*)"', String.Double), ('[^\\s"\\\\]+', Text)]}