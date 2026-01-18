from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Comment, Keyword, Name, Number, Operator, \
class WhileyLexer(RegexLexer):
    """
    Lexer for the Whiley programming language.

    .. versionadded:: 2.2
    """
    name = 'Whiley'
    filenames = ['*.whiley']
    aliases = ['whiley']
    mimetypes = ['text/x-whiley']
    tokens = {'root': [('\\s+', Text), ('//.*', Comment.Single), ('/\\*\\*/', Comment.Multiline), ('(?s)/\\*\\*.*?\\*/', String.Doc), ('(?s)/\\*.*?\\*/', Comment.Multiline), (words(('if', 'else', 'while', 'for', 'do', 'return', 'switch', 'case', 'default', 'break', 'continue', 'requires', 'ensures', 'where', 'assert', 'assume', 'all', 'no', 'some', 'in', 'is', 'new', 'throw', 'try', 'catch', 'debug', 'skip', 'fail', 'finite', 'total'), suffix='\\b'), Keyword.Reserved), (words(('function', 'method', 'public', 'private', 'protected', 'export', 'native'), suffix='\\b'), Keyword.Declaration), ('(constant|type)(\\s+)([a-zA-Z_]\\w*)(\\s+)(is)\\b', bygroups(Keyword.Declaration, Text, Name, Text, Keyword.Reserved)), ('(true|false|null)\\b', Keyword.Constant), ('(bool|byte|int|real|any|void)\\b', Keyword.Type), ('(import)(\\s+)(\\*)([^\\S\\n]+)(from)\\b', bygroups(Keyword.Namespace, Text, Punctuation, Text, Keyword.Namespace)), ('(import)(\\s+)([a-zA-Z_]\\w*)([^\\S\\n]+)(from)\\b', bygroups(Keyword.Namespace, Text, Name, Text, Keyword.Namespace)), ('(package|import)\\b', Keyword.Namespace), (words(('i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'uint', 'nat', 'toString'), suffix='\\b'), Name.Builtin), ('[01]+b', Number.Bin), ('[0-9]+\\.[0-9]+', Number.Float), ('[0-9]+\\.(?!\\.)', Number.Float), ('0x[0-9a-fA-F]+', Number.Hex), ('[0-9]+', Number.Integer), ("'[^\\\\]'", String.Char), ('(\')(\\\\[\'"\\\\btnfr])(\')', bygroups(String.Char, String.Escape, String.Char)), ('"', String, 'string'), ('[{}()\\[\\],.;]', Punctuation), (u'[+\\-*/%&|<>^!~@=:?∀∃∅⊂⊆⊃⊇∪∩≤≥∈∧∨]', Operator), ('[a-zA-Z_]\\w*', Name)], 'string': [('"', String, '#pop'), ('\\\\[btnfr]', String.Escape), ('\\\\u[0-9a-fA-F]{4}', String.Escape), ('\\\\.', String), ('[^\\\\"]+', String)]}