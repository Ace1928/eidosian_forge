import re
from pygments.lexer import RegexLexer, bygroups, default, words, include
from pygments.token import Comment, Error, Keyword, Name, Number, \
from pygments.lexers import _vbscript_builtins
class VBScriptLexer(RegexLexer):
    """
    VBScript is scripting language that is modeled on Visual Basic.

    .. versionadded:: 2.4
    """
    name = 'VBScript'
    aliases = ['vbscript']
    filenames = ['*.vbs', '*.VBS']
    flags = re.IGNORECASE
    tokens = {'root': [("'[^\\n]*", Comment.Single), ('\\s+', Whitespace), ('"', String.Double, 'string'), ('&h[0-9a-f]+', Number.Hex), ('[0-9]+\\.[0-9]*(e[+-]?[0-9]+)?', Number.Float), ('\\.[0-9]+(e[+-]?[0-9]+)?', Number.Float), ('[0-9]+e[+-]?[0-9]+', Number.Float), ('[0-9]+', Number.Integer), ('#.+#', String), ('(dim)(\\s+)([a-z_][a-z0-9_]*)', bygroups(Keyword.Declaration, Whitespace, Name.Variable), 'dim_more'), ('(function|sub)(\\s+)([a-z_][a-z0-9_]*)', bygroups(Keyword.Declaration, Whitespace, Name.Function)), ('(class)(\\s+)([a-z_][a-z0-9_]*)', bygroups(Keyword.Declaration, Whitespace, Name.Class)), ('(const)(\\s+)([a-z_][a-z0-9_]*)', bygroups(Keyword.Declaration, Whitespace, Name.Constant)), ('(end)(\\s+)(class|function|if|property|sub|with)', bygroups(Keyword, Whitespace, Keyword)), ('(on)(\\s+)(error)(\\s+)(goto)(\\s+)(0)', bygroups(Keyword, Whitespace, Keyword, Whitespace, Keyword, Whitespace, Number.Integer)), ('(on)(\\s+)(error)(\\s+)(resume)(\\s+)(next)', bygroups(Keyword, Whitespace, Keyword, Whitespace, Keyword, Whitespace, Keyword)), ('(option)(\\s+)(explicit)', bygroups(Keyword, Whitespace, Keyword)), ('(property)(\\s+)(get|let|set)(\\s+)([a-z_][a-z0-9_]*)', bygroups(Keyword.Declaration, Whitespace, Keyword.Declaration, Whitespace, Name.Property)), ('rem\\s.*[^\\n]*', Comment.Single), (words(_vbscript_builtins.KEYWORDS, suffix='\\b'), Keyword), (words(_vbscript_builtins.OPERATORS), Operator), (words(_vbscript_builtins.OPERATOR_WORDS, suffix='\\b'), Operator.Word), (words(_vbscript_builtins.BUILTIN_CONSTANTS, suffix='\\b'), Name.Constant), (words(_vbscript_builtins.BUILTIN_FUNCTIONS, suffix='\\b'), Name.Builtin), (words(_vbscript_builtins.BUILTIN_VARIABLES, suffix='\\b'), Name.Builtin), ('[a-z_][a-z0-9_]*', Name), ('\\b_\\n', Operator), (words('(),.:'), Punctuation), ('.+(\\n)?', Error)], 'dim_more': [('(\\s*)(,)(\\s*)([a-z_][a-z0-9]*)', bygroups(Whitespace, Punctuation, Whitespace, Name.Variable)), default('#pop')], 'string': [('[^"\\n]+', String.Double), ('\\"\\"', String.Double), ('"', String.Double, '#pop'), ('\\n', Error, '#pop')]}