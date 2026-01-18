from pygments.lexer import ExtendedRegexLexer, bygroups, DelegatingLexer
from pygments.token import Name, Number, String, Comment, Punctuation, \
class SlashLexer(DelegatingLexer):
    """
    Lexer for the Slash programming language.

    .. versionadded:: 2.4
    """
    name = 'Slash'
    aliases = ['slash']
    filenames = ['*.sla']

    def __init__(self, **options):
        from pygments.lexers.web import HtmlLexer
        super().__init__(HtmlLexer, SlashLanguageLexer, **options)