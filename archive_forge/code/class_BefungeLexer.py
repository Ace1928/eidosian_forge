from pygments.lexer import RegexLexer, include, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class BefungeLexer(RegexLexer):
    """
    Lexer for the esoteric `Befunge <http://en.wikipedia.org/wiki/Befunge>`_
    language.

    .. versionadded:: 0.7
    """
    name = 'Befunge'
    aliases = ['befunge']
    filenames = ['*.befunge']
    mimetypes = ['application/x-befunge']
    tokens = {'root': [('[0-9a-f]', Number), ('[+*/%!`-]', Operator), ('[<>^v?\\[\\]rxjk]', Name.Variable), ('[:\\\\$.,n]', Name.Builtin), ('[|_mw]', Keyword), ('[{}]', Name.Tag), ('".*?"', String.Double), ("\\'.", String.Single), ('[#;]', Comment), ('[pg&~=@iotsy]', Keyword), ('[()A-Z]', Comment), ('\\s+', Text)]}