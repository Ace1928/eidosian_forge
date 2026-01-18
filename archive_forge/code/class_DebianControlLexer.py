import re
from pygments.lexer import RegexLexer, include, bygroups, using, this, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class DebianControlLexer(RegexLexer):
    """
    Lexer for Debian ``control`` files and ``apt-cache show <pkg>`` outputs.

    .. versionadded:: 0.9
    """
    name = 'Debian Control file'
    aliases = ['control', 'debcontrol']
    filenames = ['control']
    tokens = {'root': [('^(Description)', Keyword, 'description'), ('^(Maintainer)(:\\s*)', bygroups(Keyword, Text), 'maintainer'), ('^((Build-)?Depends)', Keyword, 'depends'), ('^((?:Python-)?Version)(:\\s*)(\\S+)$', bygroups(Keyword, Text, Number)), ('^((?:Installed-)?Size)(:\\s*)(\\S+)$', bygroups(Keyword, Text, Number)), ('^(MD5Sum|SHA1|SHA256)(:\\s*)(\\S+)$', bygroups(Keyword, Text, Number)), ('^([a-zA-Z\\-0-9\\.]*?)(:\\s*)(.*?)$', bygroups(Keyword, Whitespace, String))], 'maintainer': [('<[^>]+>', Generic.Strong), ('<[^>]+>$', Generic.Strong, '#pop'), (',\\n?', Text), ('.', Text)], 'description': [('(.*)(Homepage)(: )(\\S+)', bygroups(Text, String, Name, Name.Class)), (':.*\\n', Generic.Strong), (' .*\\n', Text), default('#pop')], 'depends': [(':\\s*', Text), ('(\\$)(\\{)(\\w+\\s*:\\s*\\w+)', bygroups(Operator, Text, Name.Entity)), ('\\(', Text, 'depend_vers'), (',', Text), ('\\|', Operator), ('[\\s]+', Text), ('[})]\\s*$', Text, '#pop'), ('\\}', Text), ('[^,]$', Name.Function, '#pop'), ('([+.a-zA-Z0-9-])(\\s*)', bygroups(Name.Function, Text)), ('\\[.*?\\]', Name.Entity)], 'depend_vers': [('\\),', Text, '#pop'), ('\\)[^,]', Text, '#pop:2'), ('([><=]+)(\\s*)([^)]+)', bygroups(Operator, Text, Number))]}