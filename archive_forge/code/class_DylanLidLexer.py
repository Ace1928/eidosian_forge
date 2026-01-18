import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class DylanLidLexer(RegexLexer):
    """
    For Dylan LID (Library Interchange Definition) files.

    .. versionadded:: 1.6
    """
    name = 'DylanLID'
    aliases = ['dylan-lid', 'lid']
    filenames = ['*.lid', '*.hdp']
    mimetypes = ['text/x-dylan-lid']
    flags = re.IGNORECASE
    tokens = {'root': [('\\s+', Text), ('//.*?\\n', Comment.Single), ('(.*?)(:)([ \\t]*)(.*(?:\\n[ \\t].+)*)', bygroups(Name.Attribute, Operator, Text, String))]}