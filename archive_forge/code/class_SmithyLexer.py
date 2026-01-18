from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Keyword, Name, String, \
class SmithyLexer(RegexLexer):
    """
    For Smithy IDL

    .. versionadded:: 2.10
    """
    name = 'Smithy'
    url = 'https://awslabs.github.io/smithy/'
    filenames = ['*.smithy']
    aliases = ['smithy']
    unquoted = '[A-Za-z0-9_\\.#$-]+'
    identifier = '[A-Za-z0-9_\\.#$-]+'
    simple_shapes = ('use', 'byte', 'short', 'integer', 'long', 'float', 'document', 'double', 'bigInteger', 'bigDecimal', 'boolean', 'blob', 'string', 'timestamp')
    aggregate_shapes = ('apply', 'list', 'map', 'set', 'structure', 'union', 'resource', 'operation', 'service', 'trait')
    tokens = {'root': [('///.*$', Comment.Multiline), ('//.*$', Comment), ('@[0-9a-zA-Z\\.#-]*', Name.Decorator), ('(=)', Name.Decorator), ('^(\\$version)(:)(.+)', bygroups(Keyword.Declaration, Name.Decorator, Name.Class)), ('^(namespace)(\\s+' + identifier + ')\\b', bygroups(Keyword.Declaration, Name.Class)), (words(simple_shapes, prefix='^', suffix='(\\s+' + identifier + ')\\b'), bygroups(Keyword.Declaration, Name.Class)), (words(aggregate_shapes, prefix='^', suffix='(\\s+' + identifier + ')'), bygroups(Keyword.Declaration, Name.Class)), ('^(metadata)(\\s+)((?:\\S+)|(?:\\"[^"]+\\"))(\\s*)(=)', bygroups(Keyword.Declaration, Whitespace, Name.Class, Whitespace, Name.Decorator)), ('(true|false|null)', Keyword.Constant), ('(-?(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)', Number), (identifier + ':', Name.Label), (identifier, Name.Variable.Class), ('\\[', Text, '#push'), ('\\]', Text, '#pop'), ('\\(', Text, '#push'), ('\\)', Text, '#pop'), ('\\{', Text, '#push'), ('\\}', Text, '#pop'), ('"{3}(\\\\\\\\|\\n|\\\\")*"{3}', String.Doc), ('"(\\\\\\\\|\\n|\\\\"|[^"])*"', String.Double), ("'(\\\\\\\\|\\n|\\\\'|[^'])*'", String.Single), ('[:,]+', Punctuation), ('\\s+', Whitespace)]}