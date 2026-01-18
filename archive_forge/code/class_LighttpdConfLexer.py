import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class LighttpdConfLexer(RegexLexer):
    """
    Lexer for `Lighttpd <http://lighttpd.net/>`_ configuration files.

    .. versionadded:: 0.11
    """
    name = 'Lighttpd configuration file'
    aliases = ['lighty', 'lighttpd']
    filenames = []
    mimetypes = ['text/x-lighttpd-conf']
    tokens = {'root': [('#.*\\n', Comment.Single), ('/\\S*', Name), ('[a-zA-Z._-]+', Keyword), ('\\d+\\.\\d+\\.\\d+\\.\\d+(?:/\\d+)?', Number), ('[0-9]+', Number), ('=>|=~|\\+=|==|=|\\+', Operator), ('\\$[A-Z]+', Name.Builtin), ('[(){}\\[\\],]', Punctuation), ('"([^"\\\\]*(?:\\\\.[^"\\\\]*)*)"', String.Double), ('\\s+', Text)]}