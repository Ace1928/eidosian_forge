import re
from pygments.lexer import RegexLexer, include, bygroups, words
from pygments.token import Comment, Keyword, Name, Number, Operator, \
from pygments.lexers._qlik_builtins import OPERATORS_LIST, STATEMENT_LIST, \
class QlikLexer(RegexLexer):
    """
    Lexer for qlik code, including .qvs files

    .. versionadded:: 2.12
    """
    name = 'Qlik'
    aliases = ['qlik', 'qlikview', 'qliksense', 'qlikscript']
    filenames = ['*.qvs', '*.qvw']
    flags = re.IGNORECASE
    tokens = {'comment': [('\\*/', Comment.Multiline, '#pop'), ('[^*]+', Comment.Multiline)], 'numerics': [('\\b\\d+\\.\\d+(e\\d+)?[fd]?\\b', Number.Float), ('\\b\\d+\\b', Number.Integer)], 'interp': [('(\\$\\()(\\w+)(\\))', bygroups(String.Interpol, Name.Variable, String.Interpol))], 'string': [("'", String, '#pop'), include('interp'), ("[^'$]+", String), ('\\$', String)], 'assignment': [(';', Punctuation, '#pop'), include('root')], 'field_name_quote': [('"', String.Symbol, '#pop'), include('interp'), ('[^\\"$]+', String.Symbol), ('\\$', String.Symbol)], 'field_name_bracket': [('\\]', String.Symbol, '#pop'), include('interp'), ('[^\\]$]+', String.Symbol), ('\\$', String.Symbol)], 'function': [('\\)', Punctuation, '#pop'), include('root')], 'root': [('\\s+', Text.Whitespace), ('/\\*', Comment.Multiline, 'comment'), ('//.*\\n', Comment.Single), ('(let|set)(\\s+)', bygroups(Keyword.Declaration, Text.Whitespace), 'assignment'), (words(OPERATORS_LIST['words'], prefix='\\b', suffix='\\b'), Operator.Word), (words(STATEMENT_LIST, suffix='\\b'), Keyword), ('[a-z]\\w*:', Keyword.Declaration), (words(CONSTANT_LIST, suffix='\\b'), Keyword.Constant), (words(SCRIPT_FUNCTIONS, suffix='(?=\\s*\\()'), Name.Builtin, 'function'), include('interp'), ('"', String.Symbol, 'field_name_quote'), ('\\[', String.Symbol, 'field_name_bracket'), ("'", String, 'string'), include('numerics'), (words(OPERATORS_LIST['symbols']), Operator), ("'.+?'", String), ('\\b\\w+\\b', Text), ('[,;.()\\\\/]', Punctuation)]}