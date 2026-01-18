import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class IniLexer(RegexLexer):
    """
    Lexer for configuration files in INI style.
    """
    name = 'INI'
    aliases = ['ini', 'cfg', 'dosini']
    filenames = ['*.ini', '*.cfg', '*.inf']
    mimetypes = ['text/x-ini', 'text/inf']
    tokens = {'root': [('\\s+', Text), ('[;#].*', Comment.Single), ('\\[.*?\\]$', Keyword), ('(.*?)([ \\t]*)(=)([ \\t]*)(.*(?:\\n[ \\t].+)*)', bygroups(Name.Attribute, Text, Operator, Text, String)), ('(.+?)$', Name.Attribute)]}

    def analyse_text(text):
        npos = text.find('\n')
        if npos < 3:
            return False
        return text[0] == '[' and text[npos - 1] == ']'