import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class NginxConfLexer(RegexLexer):
    """
    Lexer for `Nginx <http://nginx.net/>`_ configuration files.

    .. versionadded:: 0.11
    """
    name = 'Nginx configuration file'
    aliases = ['nginx']
    filenames = ['nginx.conf']
    mimetypes = ['text/x-nginx-conf']
    tokens = {'root': [('(include)(\\s+)([^\\s;]+)', bygroups(Keyword, Text, Name)), ('[^\\s;#]+', Keyword, 'stmt'), include('base')], 'block': [('\\}', Punctuation, '#pop:2'), ('[^\\s;#]+', Keyword.Namespace, 'stmt'), include('base')], 'stmt': [('\\{', Punctuation, 'block'), (';', Punctuation, '#pop'), include('base')], 'base': [('#.*\\n', Comment.Single), ('on|off', Name.Constant), ('\\$[^\\s;#()]+', Name.Variable), ('([a-z0-9.-]+)(:)([0-9]+)', bygroups(Name, Punctuation, Number.Integer)), ('[a-z-]+/[a-z-+]+', String), ('[0-9]+[km]?\\b', Number.Integer), ('(~)(\\s*)([^\\s{]+)', bygroups(Punctuation, Text, String.Regex)), ('[:=~]', Punctuation), ('[^\\s;#{}$]+', String), ('/[^\\s;#]*', Name), ('\\s+', Text), ('[$;]', Text)]}