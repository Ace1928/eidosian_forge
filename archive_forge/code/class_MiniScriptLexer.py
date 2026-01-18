import re
from pygments.lexer import RegexLexer, include, bygroups, default, combined, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, get_list_opt
class MiniScriptLexer(RegexLexer):
    """
    For MiniScript source code.

    .. versionadded:: 2.6
    """
    name = 'MiniScript'
    url = 'https://miniscript.org'
    aliases = ['miniscript', 'ms']
    filenames = ['*.ms']
    mimetypes = ['text/x-minicript', 'application/x-miniscript']
    tokens = {'root': [('#!(.*?)$', Comment.Preproc), default('base')], 'base': [('//.*$', Comment.Single), ('(?i)(\\d*\\.\\d+|\\d+\\.\\d*)(e[+-]?\\d+)?', Number), ('(?i)\\d+e[+-]?\\d+', Number), ('\\d+', Number), ('\\n', Text), ('[^\\S\\n]+', Text), ('"', String, 'string_double'), ('(==|!=|<=|>=|[=+\\-*/%^<>.:])', Operator), ('[;,\\[\\]{}()]', Punctuation), (words(('break', 'continue', 'else', 'end', 'for', 'function', 'if', 'in', 'isa', 'then', 'repeat', 'return', 'while'), suffix='\\b'), Keyword), (words(('abs', 'acos', 'asin', 'atan', 'ceil', 'char', 'cos', 'floor', 'log', 'round', 'rnd', 'pi', 'sign', 'sin', 'sqrt', 'str', 'tan', 'hasIndex', 'indexOf', 'len', 'val', 'code', 'remove', 'lower', 'upper', 'replace', 'split', 'indexes', 'values', 'join', 'sum', 'sort', 'shuffle', 'push', 'pop', 'pull', 'range', 'print', 'input', 'time', 'wait', 'locals', 'globals', 'outer', 'yield'), suffix='\\b'), Name.Builtin), ('(true|false|null)\\b', Keyword.Constant), ('(and|or|not|new)\\b', Operator.Word), ('(self|super|__isa)\\b', Name.Builtin.Pseudo), ('[a-zA-Z_]\\w*', Name.Variable)], 'string_double': [('[^"\\n]+', String), ('""', String), ('"', String, '#pop'), ('\\n', Text, '#pop')]}