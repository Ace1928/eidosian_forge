import re
from pygments.lexer import Lexer, RegexLexer, bygroups, words, do_insertions
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers import _scilab_builtins
class ScilabLexer(RegexLexer):
    """
    For Scilab source code.

    .. versionadded:: 1.5
    """
    name = 'Scilab'
    aliases = ['scilab']
    filenames = ['*.sci', '*.sce', '*.tst']
    mimetypes = ['text/scilab']
    tokens = {'root': [('//.*?$', Comment.Single), ('^\\s*function', Keyword, 'deffunc'), (words(('__FILE__', '__LINE__', 'break', 'case', 'catch', 'classdef', 'continue', 'do', 'else', 'elseif', 'end', 'end_try_catch', 'end_unwind_protect', 'endclassdef', 'endevents', 'endfor', 'endfunction', 'endif', 'endmethods', 'endproperties', 'endswitch', 'endwhile', 'events', 'for', 'function', 'get', 'global', 'if', 'methods', 'otherwise', 'persistent', 'properties', 'return', 'set', 'static', 'switch', 'try', 'until', 'unwind_protect', 'unwind_protect_cleanup', 'while'), suffix='\\b'), Keyword), (words(_scilab_builtins.functions_kw + _scilab_builtins.commands_kw + _scilab_builtins.macros_kw, suffix='\\b'), Name.Builtin), (words(_scilab_builtins.variables_kw, suffix='\\b'), Name.Constant), ('-|==|~=|<|>|<=|>=|&&|&|~|\\|\\|?', Operator), ('\\.\\*|\\*|\\+|\\.\\^|\\.\\\\|\\.\\/|\\/|\\\\', Operator), ('[\\[\\](){}@.,=:;]', Punctuation), ('"[^"]*"', String), ("(?<=[\\w)\\].])\\'+", Operator), ("(?<![\\w)\\].])\\'", String, 'string'), ('(\\d+\\.\\d*|\\d*\\.\\d+)([eEf][+-]?[0-9]+)?', Number.Float), ('\\d+[eEf][+-]?[0-9]+', Number.Float), ('\\d+', Number.Integer), ('[a-zA-Z_]\\w*', Name), ('.', Text)], 'string': [("[^']*'", String, '#pop'), ('.', String, '#pop')], 'deffunc': [('(\\s*)(?:(.+)(\\s*)(=)(\\s*))?(.+)(\\()(.*)(\\))(\\s*)', bygroups(Whitespace, Text, Whitespace, Punctuation, Whitespace, Name.Function, Punctuation, Text, Punctuation, Whitespace), '#pop'), ('(\\s*)([a-zA-Z_]\\w*)', bygroups(Text, Name.Function), '#pop')]}