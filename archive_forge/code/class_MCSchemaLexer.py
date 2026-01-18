from pygments.lexer import RegexLexer, default, include, bygroups
from pygments.token import Comment, Keyword, Literal, Name, Number, Operator, \
class MCSchemaLexer(RegexLexer):
    """Lexer for Minecraft Add-ons data Schemas, an interface structure standard used in Minecraft

    .. versionadded:: 2.14.0
    """
    name = 'MCSchema'
    url = 'https://learn.microsoft.com/en-us/minecraft/creator/reference/content/schemasreference/'
    aliases = ['mcschema']
    filenames = ['*.mcschema']
    mimetypes = ['text/mcschema']
    tokens = {'commentsandwhitespace': [('\\s+', Whitespace), ('//.*?$', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline)], 'slashstartsregex': [include('commentsandwhitespace'), ('/(\\\\.|[^[/\\\\\\n]|\\[(\\\\.|[^\\]\\\\\\n])*])+/([gimuysd]+\\b|\\B)', String.Regex, '#pop'), ('(?=/)', Text, ('#pop', 'badregex')), default('#pop')], 'badregex': [('\\n', Whitespace, '#pop')], 'singlestring': [('\\\\.', String.Escape), ("'", String.Single, '#pop'), ("[^\\\\']+", String.Single)], 'doublestring': [('\\\\.', String.Escape), ('"', String.Double, '#pop'), ('[^\\\\"]+', String.Double)], 'root': [('^(?=\\s|/|<!--)', Text, 'slashstartsregex'), include('commentsandwhitespace'), ('(?<=: )opt', Operator.Word), ('(?<=\\s)[\\w-]*(?=(\\s+"|\\n))', Keyword.Declaration), ('0[bB][01]+', Number.Bin), ('0[oO]?[0-7]+', Number.Oct), ('0[xX][0-9a-fA-F]+', Number.Hex), ('\\d+', Number.Integer), ('(\\.\\d+|\\d+\\.\\d*|\\d+)([eE][-+]?\\d+)?', Number.Float), ('\\.\\.\\.|=>', Punctuation), ('\\+\\+|--|~|\\?\\?=?|\\?|:|\\\\(?=\\n)|(<<|>>>?|==?|!=?|(?:\\*\\*|\\|\\||&&|[-<>+*%&|^/]))=?', Operator, 'slashstartsregex'), ('[{(\\[;,]', Punctuation, 'slashstartsregex'), ('[})\\].]', Punctuation), ("'", String.Single, 'singlestring'), ('"', String.Double, 'doublestring'), ('[\\w-]*?(?=:\\{?\\n)', String.Symbol), ('([\\w-]*?)(:)(\\d+)(?:(\\.)(\\d+)(?:(\\.)(\\d+)(?:(\\-)((?:[^\\W_]|-)*(?:\\.(?:[^\\W_]|-)*)*))?(?:(\\+)((?:[^\\W_]|-)+(?:\\.(?:[^\\W_]|-)+)*))?)?)?(?=:\\{?\\n)', bygroups(String.Symbol, Operator, Number.Integer, Operator, Number.Integer, Operator, Number.Integer, Operator, String, Operator, String)), ('.*\\n', Text)]}