from pygments.lexer import RegexLexer, bygroups, include, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class SolidityLexer(RegexLexer):
    """
    For Solidity source code.

    .. versionadded:: 2.5
    """
    name = 'Solidity'
    aliases = ['solidity']
    filenames = ['*.sol']
    mimetypes = []
    datatype = '\\b(address|bool|(?:(?:bytes|hash|int|string|uint)(?:8|16|24|32|40|48|56|64|72|80|88|96|104|112|120|128|136|144|152|160|168|176|184|192|200|208|216|224|232|240|248|256)?))\\b'
    tokens = {'root': [include('whitespace'), include('comments'), ('\\bpragma\\s+solidity\\b', Keyword, 'pragma'), ('\\b(contract)(\\s+)([a-zA-Z_]\\w*)', bygroups(Keyword, Whitespace, Name.Entity)), (datatype + '(\\s+)((?:external|public|internal|private)\\s+)?' + '([a-zA-Z_]\\w*)', bygroups(Keyword.Type, Whitespace, Keyword, Name.Variable)), ('\\b(enum|event|function|struct)(\\s+)([a-zA-Z_]\\w*)', bygroups(Keyword.Type, Whitespace, Name.Variable)), ('\\b(msg|block|tx)\\.([A-Za-z_][a-zA-Z0-9_]*)\\b', Keyword), (words(('block', 'break', 'constant', 'constructor', 'continue', 'contract', 'do', 'else', 'external', 'false', 'for', 'function', 'if', 'import', 'inherited', 'internal', 'is', 'library', 'mapping', 'memory', 'modifier', 'msg', 'new', 'payable', 'private', 'public', 'require', 'return', 'returns', 'struct', 'suicide', 'throw', 'this', 'true', 'tx', 'var', 'while'), prefix='\\b', suffix='\\b'), Keyword.Type), (words(('keccak256',), prefix='\\b', suffix='\\b'), Name.Builtin), (datatype, Keyword.Type), include('constants'), ('[a-zA-Z_]\\w*', Text), ('[~!%^&*+=|?:<>/-]', Operator), ('[.;{}(),\\[\\]]', Punctuation)], 'comments': [('//(\\n|[\\w\\W]*?[^\\\\]\\n)', Comment.Single), ('/(\\\\\\n)?[*][\\w\\W]*?[*](\\\\\\n)?/', Comment.Multiline), ('/(\\\\\\n)?[*][\\w\\W]*', Comment.Multiline)], 'constants': [('("(\\\\"|.)*?")', String.Double), ("('(\\\\'|.)*?')", String.Single), ('\\b0[xX][0-9a-fA-F]+\\b', Number.Hex), ('\\b\\d+\\b', Number.Decimal)], 'pragma': [include('whitespace'), include('comments'), ('(\\^|>=|<)(\\s*)(\\d+\\.\\d+\\.\\d+)', bygroups(Operator, Whitespace, Keyword)), (';', Punctuation, '#pop')], 'whitespace': [('\\s+', Whitespace), ('\\n', Whitespace)]}