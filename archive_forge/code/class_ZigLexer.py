from pygments.lexer import RegexLexer, words
from pygments.token import Comment, Operator, Keyword, Name, String, \
class ZigLexer(RegexLexer):
    """
    Lexer for the Zig language.

    grammar: https://ziglang.org/documentation/master/#Grammar
    """
    name = 'Zig'
    url = 'http://www.ziglang.org'
    aliases = ['zig']
    filenames = ['*.zig']
    mimetypes = ['text/zig']
    type_keywords = (words(('bool', 'f16', 'f32', 'f64', 'f128', 'void', 'noreturn', 'type', 'anyerror', 'promise', 'i0', 'u0', 'isize', 'usize', 'comptime_int', 'comptime_float', 'c_short', 'c_ushort', 'c_int', 'c_uint', 'c_long', 'c_ulong', 'c_longlong', 'c_ulonglong', 'c_longdouble', 'c_voidi8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'i128', 'u128'), suffix='\\b'), Keyword.Type)
    storage_keywords = (words(('const', 'var', 'extern', 'packed', 'export', 'pub', 'noalias', 'inline', 'comptime', 'nakedcc', 'stdcallcc', 'volatile', 'allowzero', 'align', 'linksection', 'threadlocal'), suffix='\\b'), Keyword.Reserved)
    structure_keywords = (words(('struct', 'enum', 'union', 'error'), suffix='\\b'), Keyword)
    statement_keywords = (words(('break', 'return', 'continue', 'asm', 'defer', 'errdefer', 'unreachable', 'try', 'catch', 'async', 'await', 'suspend', 'resume', 'cancel'), suffix='\\b'), Keyword)
    conditional_keywords = (words(('if', 'else', 'switch', 'and', 'or', 'orelse'), suffix='\\b'), Keyword)
    repeat_keywords = (words(('while', 'for'), suffix='\\b'), Keyword)
    other_keywords = (words(('fn', 'usingnamespace', 'test'), suffix='\\b'), Keyword)
    constant_keywords = (words(('true', 'false', 'null', 'undefined'), suffix='\\b'), Keyword.Constant)
    tokens = {'root': [('\\n', Whitespace), ('\\s+', Whitespace), ('//.*?\\n', Comment.Single), statement_keywords, storage_keywords, structure_keywords, repeat_keywords, type_keywords, constant_keywords, conditional_keywords, other_keywords, ('0x[0-9a-fA-F]+\\.[0-9a-fA-F]+([pP][\\-+]?[0-9a-fA-F]+)?', Number.Float), ('0x[0-9a-fA-F]+\\.?[pP][\\-+]?[0-9a-fA-F]+', Number.Float), ('[0-9]+\\.[0-9]+([eE][-+]?[0-9]+)?', Number.Float), ('[0-9]+\\.?[eE][-+]?[0-9]+', Number.Float), ('0b[01]+', Number.Bin), ('0o[0-7]+', Number.Oct), ('0x[0-9a-fA-F]+', Number.Hex), ('[0-9]+', Number.Integer), ('@[a-zA-Z_]\\w*', Name.Builtin), ('[a-zA-Z_]\\w*', Name), ("\\'\\\\\\'\\'", String.Escape), ('\\\'\\\\(x[a-fA-F0-9]{2}|u[a-fA-F0-9]{4}|U[a-fA-F0-9]{6}|[nr\\\\t\\\'"])\\\'', String.Escape), ("\\'[^\\\\\\']\\'", String), ('\\\\\\\\[^\\n]*', String.Heredoc), ('c\\\\\\\\[^\\n]*', String.Heredoc), ('c?"', String, 'string'), ('[+%=><|^!?/\\-*&~:]', Operator), ('[{}()\\[\\],.;]', Punctuation)], 'string': [('\\\\(x[a-fA-F0-9]{2}|u[a-fA-F0-9]{4}|U[a-fA-F0-9]{6}|[nr\\\\t\\\'"])', String.Escape), ('[^\\\\"\\n]+', String), ('"', String, '#pop')]}