from pygments.lexer import RegexLexer
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class IoLexer(RegexLexer):
    """
    For `Io <http://iolanguage.com/>`_ (a small, prototype-based
    programming language) source.

    .. versionadded:: 0.10
    """
    name = 'Io'
    filenames = ['*.io']
    aliases = ['io']
    mimetypes = ['text/x-iosrc']
    tokens = {'root': [('\\n', Text), ('\\s+', Text), ('//(.*?)\\n', Comment.Single), ('#(.*?)\\n', Comment.Single), ('/(\\\\\\n)?[*](.|\\n)*?[*](\\\\\\n)?/', Comment.Multiline), ('/\\+', Comment.Multiline, 'nestedcomment'), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ('::=|:=|=|\\(|\\)|;|,|\\*|-|\\+|>|<|@|!|/|\\||\\^|\\.|%|&|\\[|\\]|\\{|\\}', Operator), ('(clone|do|doFile|doString|method|for|if|else|elseif|then)\\b', Keyword), ('(nil|false|true)\\b', Name.Constant), ('(Object|list|List|Map|args|Sequence|Coroutine|File)\\b', Name.Builtin), ('[a-zA-Z_]\\w*', Name), ('(\\d+\\.?\\d*|\\d*\\.\\d+)([eE][+-]?[0-9]+)?', Number.Float), ('\\d+', Number.Integer)], 'nestedcomment': [('[^+/]+', Comment.Multiline), ('/\\+', Comment.Multiline, '#push'), ('\\+/', Comment.Multiline, '#pop'), ('[+/]', Comment.Multiline)]}