from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class ChapelLexer(RegexLexer):
    """
    For `Chapel <http://chapel.cray.com/>`_ source.

    .. versionadded:: 2.0
    """
    name = 'Chapel'
    filenames = ['*.chpl']
    aliases = ['chapel', 'chpl']
    tokens = {'root': [('\\n', Text), ('\\s+', Text), ('\\\\\\n', Text), ('//(.*?)\\n', Comment.Single), ('/(\\\\\\n)?[*](.|\\n)*?[*](\\\\\\n)?/', Comment.Multiline), ('(config|const|in|inout|out|param|ref|type|var)\\b', Keyword.Declaration), ('(false|nil|true)\\b', Keyword.Constant), ('(bool|complex|imag|int|opaque|range|real|string|uint)\\b', Keyword.Type), (words(('align', 'as', 'atomic', 'begin', 'break', 'by', 'cobegin', 'coforall', 'continue', 'delete', 'dmapped', 'do', 'domain', 'else', 'enum', 'except', 'export', 'extern', 'for', 'forall', 'if', 'index', 'inline', 'iter', 'label', 'lambda', 'let', 'local', 'new', 'noinit', 'on', 'only', 'otherwise', 'pragma', 'private', 'public', 'reduce', 'require', 'return', 'scan', 'select', 'serial', 'single', 'sparse', 'subdomain', 'sync', 'then', 'use', 'when', 'where', 'while', 'with', 'yield', 'zip'), suffix='\\b'), Keyword), ('(proc)((?:\\s)+)', bygroups(Keyword, Text), 'procname'), ('(class|module|record|union)(\\s+)', bygroups(Keyword, Text), 'classname'), ('\\d+i', Number), ('\\d+\\.\\d*([Ee][-+]\\d+)?i', Number), ('\\.\\d+([Ee][-+]\\d+)?i', Number), ('\\d+[Ee][-+]\\d+i', Number), ('(\\d*\\.\\d+)([eE][+-]?[0-9]+)?i?', Number.Float), ('\\d+[eE][+-]?[0-9]+i?', Number.Float), ('0[bB][01]+', Number.Bin), ('0[xX][0-9a-fA-F]+', Number.Hex), ('0[oO][0-7]+', Number.Oct), ('[0-9]+', Number.Integer), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ("'(\\\\\\\\|\\\\'|[^'])*'", String), ('(=|\\+=|-=|\\*=|/=|\\*\\*=|%=|&=|\\|=|\\^=|&&=|\\|\\|=|<<=|>>=|<=>|<~>|\\.\\.|by|#|\\.\\.\\.|&&|\\|\\||!|&|\\||\\^|~|<<|>>|==|!=|<=|>=|<|>|[+\\-*/%]|\\*\\*)', Operator), ('[:;,.?()\\[\\]{}]', Punctuation), ('[a-zA-Z_][\\w$]*', Name.Other)], 'classname': [('[a-zA-Z_][\\w$]*', Name.Class, '#pop')], 'procname': [('([a-zA-Z_][\\w$]+|\\~[a-zA-Z_][\\w$]+|[+*/!~%<>=&^|\\-]{1,2})', Name.Function, '#pop')]}