from pygments.lexer import RegexLexer, include, default, words
from pygments.token import Comment, Keyword, Name, Number, Operator, \
class SophiaLexer(RegexLexer):
    """
    A Sophia lexer.

    .. versionadded:: 2.11
    """
    name = 'Sophia'
    aliases = ['sophia']
    filenames = ['*.aes']
    mimetypes = []
    keywords = ('contract', 'include', 'let', 'switch', 'type', 'record', 'datatype', 'if', 'elif', 'else', 'function', 'stateful', 'payable', 'public', 'entrypoint', 'private', 'indexed', 'namespace', 'interface', 'main', 'using', 'as', 'for', 'hiding')
    builtins = ('state', 'put', 'abort', 'require')
    word_operators = ('mod', 'band', 'bor', 'bxor', 'bnot')
    primitive_types = ('int', 'address', 'bool', 'bits', 'bytes', 'string', 'list', 'option', 'char', 'unit', 'map', 'event', 'hash', 'signature', 'oracle', 'oracle_query')
    tokens = {'escape-sequence': [('\\\\[\\\\"\\\'ntbr]', String.Escape), ('\\\\[0-9]{3}', String.Escape), ('\\\\x[0-9a-fA-F]{2}', String.Escape)], 'root': [('\\s+', Text.Whitespace), ('(true|false)\\b', Keyword.Constant), ("\\b([A-Z][\\w\\']*)(?=\\s*\\.)", Name.Class, 'dotted'), ("\\b([A-Z][\\w\\']*)", Name.Function), ('//.*?\\n', Comment.Single), ('\\/\\*(?!/)', Comment.Multiline, 'comment'), ('0[xX][\\da-fA-F][\\da-fA-F_]*', Number.Hex), ('#[\\da-fA-F][\\da-fA-F_]*', Name.Label), ('\\d[\\d_]*', Number.Integer), (words(keywords, suffix='\\b'), Keyword), (words(builtins, suffix='\\b'), Name.Builtin), (words(word_operators, prefix='\\b', suffix='\\b'), Operator.Word), (words(primitive_types, prefix='\\b', suffix='\\b'), Keyword.Type), ('[=!<>+\\\\*/:&|?~@^-]', Operator.Word), ('[.;:{}(),\\[\\]]', Punctuation), ("(ak_|ok_|oq_|ct_)[\\w']*", Name.Label), ("[^\\W\\d][\\w']*", Name), ('\'(?:(\\\\[\\\\\\"\'ntbr ])|(\\\\[0-9]{3})|(\\\\x[0-9a-fA-F]{2}))\'', String.Char), ("'.'", String.Char), ("'[a-z][\\w]*", Name.Variable), ('"', String.Double, 'string')], 'comment': [('[^/*]+', Comment.Multiline), ('\\/\\*', Comment.Multiline, '#push'), ('\\*\\/', Comment.Multiline, '#pop'), ('\\*', Comment.Multiline)], 'string': [('[^\\\\"]+', String.Double), include('escape-sequence'), ('\\\\\\n', String.Double), ('"', String.Double, '#pop')], 'dotted': [('\\s+', Text), ('\\.', Punctuation), ("[A-Z][\\w\\']*(?=\\s*\\.)", Name.Function), ("[A-Z][\\w\\']*", Name.Function, '#pop'), ("[a-z_][\\w\\']*", Name, '#pop'), default('#pop')]}