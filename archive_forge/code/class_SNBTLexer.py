from pygments.lexer import RegexLexer, default, include, bygroups
from pygments.token import Comment, Keyword, Literal, Name, Number, Operator, \
class SNBTLexer(RegexLexer):
    """Lexer for stringified NBT, a data format used in Minecraft

    .. versionadded:: 2.12.0
    """
    name = 'SNBT'
    url = 'https://minecraft.fandom.com/wiki/NBT_format'
    aliases = ['snbt']
    filenames = ['*.snbt']
    mimetypes = ['text/snbt']
    tokens = {'root': [('\\{', Punctuation, 'compound'), ('[^\\{]+', Text)], 'whitespace': [('\\s+', Whitespace)], 'operators': [('[,:;]', Punctuation)], 'literals': [('(true|false)', Keyword.Constant), ('-?\\d+[eE]-?\\d+', Number.Float), ('-?\\d*\\.\\d+[fFdD]?', Number.Float), ('-?\\d+[bBsSlLfFdD]?', Number.Integer), ('"', String.Double, 'literals.string_double'), ("'", String.Single, 'literals.string_single')], 'literals.string_double': [('\\\\.', String.Escape), ('[^\\\\"\\n]+', String.Double), ('"', String.Double, '#pop')], 'literals.string_single': [('\\\\.', String.Escape), ("[^\\\\'\\n]+", String.Single), ("'", String.Single, '#pop')], 'compound': [('[A-Z_a-z]+', Name.Attribute), include('operators'), include('whitespace'), include('literals'), ('\\{', Punctuation, '#push'), ('\\[', Punctuation, 'list'), ('\\}', Punctuation, '#pop')], 'list': [('[A-Z_a-z]+', Name.Attribute), include('literals'), include('operators'), include('whitespace'), ('\\[', Punctuation, '#push'), ('\\{', Punctuation, 'compound'), ('\\]', Punctuation, '#pop')]}