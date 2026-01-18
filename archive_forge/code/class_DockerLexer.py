import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class DockerLexer(RegexLexer):
    """
    Lexer for `Docker <http://docker.io>`_ configuration files.

    .. versionadded:: 2.0
    """
    name = 'Docker'
    aliases = ['docker', 'dockerfile']
    filenames = ['Dockerfile', '*.docker']
    mimetypes = ['text/x-dockerfile-config']
    _keywords = '(?:FROM|MAINTAINER|CMD|EXPOSE|ENV|ADD|ENTRYPOINT|VOLUME|WORKDIR)'
    flags = re.IGNORECASE | re.MULTILINE
    tokens = {'root': [('^(ONBUILD)(\\s+)(%s)\\b' % (_keywords,), bygroups(Name.Keyword, Whitespace, Keyword)), ('^(%s)\\b(.*)' % (_keywords,), bygroups(Keyword, String)), ('#.*', Comment), ('RUN', Keyword), ('(.*\\\\\\n)*.+', using(BashLexer))]}