from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class PonyLexer(RegexLexer):
    """
    For Pony source code.

    .. versionadded:: 2.4
    """
    name = 'Pony'
    aliases = ['pony']
    filenames = ['*.pony']
    _caps = '(iso|trn|ref|val|box|tag)'
    tokens = {'root': [('\\n', Text), ('[^\\S\\n]+', Text), ('//.*\\n', Comment.Single), ('/\\*', Comment.Multiline, 'nested_comment'), ('"""(?:.|\\n)*?"""', String.Doc), ('"', String, 'string'), ("\\'.*\\'", String.Char), ('=>|[]{}:().~;,|&!^?[]', Punctuation), (words(('addressof', 'and', 'as', 'consume', 'digestof', 'is', 'isnt', 'not', 'or'), suffix='\\b'), Operator.Word), ('!=|==|<<|>>|[-+/*%=<>]', Operator), (words(('box', 'break', 'compile_error', 'compile_intrinsic', 'continue', 'do', 'else', 'elseif', 'embed', 'end', 'error', 'for', 'if', 'ifdef', 'in', 'iso', 'lambda', 'let', 'match', 'object', 'recover', 'ref', 'repeat', 'return', 'tag', 'then', 'this', 'trn', 'try', 'until', 'use', 'var', 'val', 'where', 'while', 'with', '#any', '#read', '#send', '#share'), suffix='\\b'), Keyword), ('(actor|class|struct|primitive|interface|trait|type)((?:\\s)+)', bygroups(Keyword, Text), 'typename'), ('(new|fun|be)((?:\\s)+)', bygroups(Keyword, Text), 'methodname'), (words(('I8', 'U8', 'I16', 'U16', 'I32', 'U32', 'I64', 'U64', 'I128', 'U128', 'ILong', 'ULong', 'ISize', 'USize', 'F32', 'F64', 'Bool', 'Pointer', 'None', 'Any', 'Array', 'String', 'Iterator'), suffix='\\b'), Name.Builtin.Type), ('_?[A-Z]\\w*', Name.Type), ('(\\d+\\.\\d*|\\.\\d+|\\d+)[eE][+-]?\\d+', Number.Float), ('0x[0-9a-fA-F]+', Number.Hex), ('\\d+', Number.Integer), ('(true|false)\\b', Name.Builtin), ('_\\d*', Name), ("_?[a-z][\\w\\']*", Name)], 'typename': [(_caps + '?((?:\\s)*)(_?[A-Z]\\w*)', bygroups(Keyword, Text, Name.Class), '#pop')], 'methodname': [(_caps + '?((?:\\s)*)(_?[a-z]\\w*)', bygroups(Keyword, Text, Name.Function), '#pop')], 'nested_comment': [('[^*/]+', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)], 'string': [('"', String, '#pop'), ('\\\\"', String), ('[^\\\\"]+', String)]}