import re
from pygments.lexer import include, RegexLexer, words
from pygments.token import Whitespace, Punctuation, Keyword, Name, Comment, \
class WrenLexer(RegexLexer):
    """
    For Wren source code, version 0.4.0.

    .. versionadded:: 2.14.0
    """
    name = 'Wren'
    url = 'https://wren.io'
    aliases = ['wren']
    filenames = ['*.wren']
    flags = re.MULTILINE | re.DOTALL
    tokens = {'root': [('\\s+', Whitespace), ('[,\\\\\\[\\]{}]', Punctuation), ('\\(', Punctuation, 'root'), ('\\)', Punctuation, '#pop'), (words(('as', 'break', 'class', 'construct', 'continue', 'else', 'for', 'foreign', 'if', 'import', 'return', 'static', 'super', 'this', 'var', 'while'), prefix='(?<!\\.)', suffix='\\b'), Keyword), (words(('true', 'false', 'null'), prefix='(?<!\\.)', suffix='\\b'), Keyword.Constant), (words(('in', 'is'), prefix='(?<!\\.)', suffix='\\b'), Operator.Word), ('/\\*', Comment.Multiline, 'comment'), ('//.*?$', Comment.Single), ('#.*?(\\(.*?\\))?$', Comment.Special), ('[!%&*+\\-./:<=>?\\\\^|~]+', Operator), ('[a-z][a-zA-Z_0-9]*', Name), ('[A-Z][a-zA-Z_0-9]*', Name.Class), ('__[a-zA-Z_0-9]*', Name.Variable.Class), ('_[a-zA-Z_0-9]*', Name.Variable.Instance), ('0x[0-9a-fA-F]+', Number.Hex), ('\\d+(\\.\\d+)?([eE][-+]?\\d+)?', Number.Float), ('""".*?"""', String), ('"', String, 'string')], 'comment': [('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('([^*/]|\\*(?!/)|/(?!\\*))+', Comment.Multiline)], 'string': [('"', String, '#pop'), ('\\\\[\\\\%"0abefnrtv]', String.Escape), ('\\\\x[a-fA-F0-9]{2}', String.Escape), ('\\\\u[a-fA-F0-9]{4}', String.Escape), ('\\\\U[a-fA-F0-9]{8}', String.Escape), ('%\\(', String.Interpol, 'interpolation'), ('[^\\\\"%]+', String)], 'interpolation': [('\\)', String.Interpol, '#pop'), include('root')]}