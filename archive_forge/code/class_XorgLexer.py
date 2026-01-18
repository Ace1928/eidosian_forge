from pygments.lexer import RegexLexer, bygroups
from pygments.token import Comment, String, Name, Text
class XorgLexer(RegexLexer):
    """Lexer for xorg.conf files."""
    name = 'Xorg'
    url = 'https://www.x.org/wiki/'
    aliases = ['xorg.conf']
    filenames = ['xorg.conf']
    mimetypes = []
    tokens = {'root': [('\\s+', Text), ('#.*$', Comment), ('((?:Sub)?Section)(\\s+)("\\w+")', bygroups(String.Escape, Text, String.Escape)), ('(End(?:Sub)?Section)', String.Escape), ('(\\w+)(\\s+)([^\\n#]+)', bygroups(Name.Builtin, Text, Name.Constant))]}