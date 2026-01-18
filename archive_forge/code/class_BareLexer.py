from pygments.lexer import RegexLexer, words, bygroups
from pygments.token import Text, Comment, Keyword, Name, Literal, Whitespace
class BareLexer(RegexLexer):
    """
    For BARE schema source.

    .. versionadded:: 2.7
    """
    name = 'BARE'
    url = 'https://baremessages.org'
    filenames = ['*.bare']
    aliases = ['bare']
    keywords = ['type', 'enum', 'u8', 'u16', 'u32', 'u64', 'uint', 'i8', 'i16', 'i32', 'i64', 'int', 'f32', 'f64', 'bool', 'void', 'data', 'string', 'optional', 'map']
    tokens = {'root': [('(type)(\\s+)([A-Z][a-zA-Z0-9]+)(\\s+)(\\{)', bygroups(Keyword, Whitespace, Name.Class, Whitespace, Text), 'struct'), ('(type)(\\s+)([A-Z][a-zA-Z0-9]+)(\\s+)(\\()', bygroups(Keyword, Whitespace, Name.Class, Whitespace, Text), 'union'), ('(type)(\\s+)([A-Z][a-zA-Z0-9]+)(\\s+)', bygroups(Keyword, Whitespace, Name, Whitespace), 'typedef'), ('(enum)(\\s+)([A-Z][a-zA-Z0-9]+)(\\s+\\{)', bygroups(Keyword, Whitespace, Name.Class, Whitespace), 'enum'), ('#.*?$', Comment), ('\\s+', Whitespace)], 'struct': [('\\{', Text, '#push'), ('\\}', Text, '#pop'), ('([a-zA-Z0-9]+)(:)(\\s*)', bygroups(Name.Attribute, Text, Whitespace), 'typedef'), ('\\s+', Whitespace)], 'union': [('\\)', Text, '#pop'), ('(\\s*)(\\|)(\\s*)', bygroups(Whitespace, Text, Whitespace)), ('[A-Z][a-zA-Z0-9]+', Name.Class), (words(keywords), Keyword), ('\\s+', Whitespace)], 'typedef': [('\\[\\]', Text), ('#.*?$', Comment, '#pop'), ('(\\[)(\\d+)(\\])', bygroups(Text, Literal, Text)), ('<|>', Text), ('\\(', Text, 'union'), ('(\\[)([a-z][a-z-A-Z0-9]+)(\\])', bygroups(Text, Keyword, Text)), ('(\\[)([A-Z][a-z-A-Z0-9]+)(\\])', bygroups(Text, Name.Class, Text)), ('([A-Z][a-z-A-Z0-9]+)', Name.Class), (words(keywords), Keyword), ('\\n', Text, '#pop'), ('\\{', Text, 'struct'), ('\\s+', Whitespace), ('\\d+', Literal)], 'enum': [('\\{', Text, '#push'), ('\\}', Text, '#pop'), ('([A-Z][A-Z0-9_]*)(\\s*=\\s*)(\\d+)', bygroups(Name.Attribute, Text, Literal)), ('([A-Z][A-Z0-9_]*)', bygroups(Name.Attribute)), ('#.*?$', Comment), ('\\s+', Whitespace)]}