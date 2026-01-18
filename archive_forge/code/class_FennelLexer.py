import re
from pygments.lexer import RegexLexer, include, bygroups, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
from pygments.lexers._scheme_builtins import scheme_keywords, scheme_builtins
class FennelLexer(RegexLexer):
    """A lexer for the Fennel programming language.

    Fennel compiles to Lua, so all the Lua builtins are recognized as well
    as the special forms that are particular to the Fennel compiler.

    .. versionadded:: 2.3
    """
    name = 'Fennel'
    url = 'https://fennel-lang.org'
    aliases = ['fennel', 'fnl']
    filenames = ['*.fnl']
    special_forms = ('#', '%', '*', '+', '-', '->', '->>', '-?>', '-?>>', '.', '..', '/', '//', ':', '<', '<=', '=', '>', '>=', '?.', '^', 'accumulate', 'and', 'band', 'bnot', 'bor', 'bxor', 'collect', 'comment', 'do', 'doc', 'doto', 'each', 'eval-compiler', 'for', 'hashfn', 'icollect', 'if', 'import-macros', 'include', 'length', 'let', 'lshift', 'lua', 'macrodebug', 'match', 'not', 'not=', 'or', 'partial', 'pick-args', 'pick-values', 'quote', 'require-macros', 'rshift', 'set', 'set-forcibly!', 'tset', 'values', 'when', 'while', 'with-open', '~=')
    declarations = ('fn', 'global', 'lambda', 'local', 'macro', 'macros', 'var', 'λ')
    builtins = ('_G', '_VERSION', 'arg', 'assert', 'bit32', 'collectgarbage', 'coroutine', 'debug', 'dofile', 'error', 'getfenv', 'getmetatable', 'io', 'ipairs', 'load', 'loadfile', 'loadstring', 'math', 'next', 'os', 'package', 'pairs', 'pcall', 'print', 'rawequal', 'rawget', 'rawlen', 'rawset', 'require', 'select', 'setfenv', 'setmetatable', 'string', 'table', 'tonumber', 'tostring', 'type', 'unpack', 'xpcall')
    valid_name = '[a-zA-Z_!$%&*+/:<=>?^~|-][\\w!$%&*+/:<=>?^~|\\.-]*'
    tokens = {'root': [(';.*$', Comment.Single), (',+', Text), ('\\s+', Whitespace), ('-?\\d+\\.\\d+', Number.Float), ('-?\\d+', Number.Integer), ('"(\\\\\\\\|\\\\[^\\\\]|[^"\\\\])*"', String), ('(true|false|nil)', Name.Constant), (':' + valid_name, String.Symbol), (words(special_forms, suffix=' '), Keyword), (words(declarations, suffix=' '), Keyword.Declaration), (words(builtins, suffix=' '), Name.Builtin), ('\\.\\.\\.', Name.Variable), (valid_name, Name.Variable), ('(\\(|\\))', Punctuation), ('(\\[|\\])', Punctuation), ('(\\{|\\})', Punctuation), ('#', Punctuation)]}