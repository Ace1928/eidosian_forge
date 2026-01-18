import re
from pygments.lexer import Lexer, RegexLexer, ExtendedRegexLexer, include, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
class FancyLexer(RegexLexer):
    """
    Pygments Lexer For `Fancy <http://www.fancy-lang.org/>`_.

    Fancy is a self-hosted, pure object-oriented, dynamic,
    class-based, concurrent general-purpose programming language
    running on Rubinius, the Ruby VM.

    .. versionadded:: 1.5
    """
    name = 'Fancy'
    filenames = ['*.fy', '*.fancypack']
    aliases = ['fancy', 'fy']
    mimetypes = ['text/x-fancysrc']
    tokens = {'balanced-regex': [('/(\\\\\\\\|\\\\/|[^/])*/[egimosx]*', String.Regex, '#pop'), ('!(\\\\\\\\|\\\\!|[^!])*![egimosx]*', String.Regex, '#pop'), ('\\\\(\\\\\\\\|[^\\\\])*\\\\[egimosx]*', String.Regex, '#pop'), ('\\{(\\\\\\\\|\\\\\\}|[^}])*\\}[egimosx]*', String.Regex, '#pop'), ('<(\\\\\\\\|\\\\>|[^>])*>[egimosx]*', String.Regex, '#pop'), ('\\[(\\\\\\\\|\\\\\\]|[^\\]])*\\][egimosx]*', String.Regex, '#pop'), ('\\((\\\\\\\\|\\\\\\)|[^)])*\\)[egimosx]*', String.Regex, '#pop'), ('@(\\\\\\\\|\\\\@|[^@])*@[egimosx]*', String.Regex, '#pop'), ('%(\\\\\\\\|\\\\%|[^%])*%[egimosx]*', String.Regex, '#pop'), ('\\$(\\\\\\\\|\\\\\\$|[^$])*\\$[egimosx]*', String.Regex, '#pop')], 'root': [('\\s+', Text), ('s\\{(\\\\\\\\|\\\\\\}|[^}])*\\}\\s*', String.Regex, 'balanced-regex'), ('s<(\\\\\\\\|\\\\>|[^>])*>\\s*', String.Regex, 'balanced-regex'), ('s\\[(\\\\\\\\|\\\\\\]|[^\\]])*\\]\\s*', String.Regex, 'balanced-regex'), ('s\\((\\\\\\\\|\\\\\\)|[^)])*\\)\\s*', String.Regex, 'balanced-regex'), ('m?/(\\\\\\\\|\\\\/|[^/\\n])*/[gcimosx]*', String.Regex), ('m(?=[/!\\\\{<\\[(@%$])', String.Regex, 'balanced-regex'), ('#(.*?)\\n', Comment.Single), ("\\'([^\\'\\s\\[\\](){}]+|\\[\\])", String.Symbol), ('"""(\\\\\\\\|\\\\"|[^"])*"""', String), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ('(def|class|try|catch|finally|retry|return|return_local|match|case|->|=>)\\b', Keyword), ('(self|super|nil|false|true)\\b', Name.Constant), ('[(){};,/?|:\\\\]', Punctuation), (words(('Object', 'Array', 'Hash', 'Directory', 'File', 'Class', 'String', 'Number', 'Enumerable', 'FancyEnumerable', 'Block', 'TrueClass', 'NilClass', 'FalseClass', 'Tuple', 'Symbol', 'Stack', 'Set', 'FancySpec', 'Method', 'Package', 'Range'), suffix='\\b'), Name.Builtin), ('[a-zA-Z](\\w|[-+?!=*/^><%])*:', Name.Function), ('[-+*/~,<>=&!?%^\\[\\].$]+', Operator), ('[A-Z]\\w*', Name.Constant), ('@[a-zA-Z_]\\w*', Name.Variable.Instance), ('@@[a-zA-Z_]\\w*', Name.Variable.Class), ('@@?', Operator), ('[a-zA-Z_]\\w*', Name), ('(0[oO]?[0-7]+(?:_[0-7]+)*)(\\s*)([/?])?', bygroups(Number.Oct, Text, Operator)), ('(0[xX][0-9A-Fa-f]+(?:_[0-9A-Fa-f]+)*)(\\s*)([/?])?', bygroups(Number.Hex, Text, Operator)), ('(0[bB][01]+(?:_[01]+)*)(\\s*)([/?])?', bygroups(Number.Bin, Text, Operator)), ('([\\d]+(?:_\\d+)*)(\\s*)([/?])?', bygroups(Number.Integer, Text, Operator)), ('\\d+([eE][+-]?[0-9]+)|\\d+\\.\\d+([eE][+-]?[0-9]+)?', Number.Float), ('\\d+', Number.Integer)]}