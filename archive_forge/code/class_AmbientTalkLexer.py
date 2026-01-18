import re
from pygments.lexer import RegexLexer, include, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class AmbientTalkLexer(RegexLexer):
    """
    Lexer for `AmbientTalk <https://code.google.com/p/ambienttalk>`_ source code.

    .. versionadded:: 2.0
    """
    name = 'AmbientTalk'
    filenames = ['*.at']
    aliases = ['at', 'ambienttalk', 'ambienttalk/2']
    mimetypes = ['text/x-ambienttalk']
    flags = re.MULTILINE | re.DOTALL
    builtin = words(('if:', 'then:', 'else:', 'when:', 'whenever:', 'discovered:', 'disconnected:', 'reconnected:', 'takenOffline:', 'becomes:', 'export:', 'as:', 'object:', 'actor:', 'mirror:', 'taggedAs:', 'mirroredBy:', 'is:'))
    tokens = {'root': [('\\s+', Text), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline), ('(def|deftype|import|alias|exclude)\\b', Keyword), (builtin, Name.Builtin), ('(true|false|nil)\\b', Keyword.Constant), ('(~|lobby|jlobby|/)\\.', Keyword.Constant, 'namespace'), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ('\\|', Punctuation, 'arglist'), ('<:|[*^!%&<>+=,./?-]|:=', Operator), ('`[a-zA-Z_]\\w*', String.Symbol), ('[a-zA-Z_]\\w*:', Name.Function), ('[{}()\\[\\];`]', Punctuation), ('(self|super)\\b', Name.Variable.Instance), ('[a-zA-Z_]\\w*', Name.Variable), ('@[a-zA-Z_]\\w*', Name.Class), ('@\\[', Name.Class, 'annotations'), include('numbers')], 'numbers': [('(\\d+\\.\\d*|\\d*\\.\\d+)([eE][+-]?[0-9]+)?', Number.Float), ('\\d+', Number.Integer)], 'namespace': [('[a-zA-Z_]\\w*\\.', Name.Namespace), ('[a-zA-Z_]\\w*:', Name.Function, '#pop'), ('[a-zA-Z_]\\w*(?!\\.)', Name.Function, '#pop')], 'annotations': [('(.*?)\\]', Name.Class, '#pop')], 'arglist': [('\\|', Punctuation, '#pop'), ('\\s*(,)\\s*', Punctuation), ('[a-zA-Z_]\\w*', Name.Variable)]}