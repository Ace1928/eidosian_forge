from pygments.lexer import RegexLexer, DelegatingLexer, bygroups
from pygments.lexers.mime import MIMELexer
from pygments.token import Text, Keyword, Name, String, Number, Comment
from pygments.util import get_bool_opt
class EmailLexer(DelegatingLexer):
    """
    Lexer for raw E-mail.

    Additional options accepted:

    `highlight-X-header`
        Highlight the fields of ``X-`` user-defined email header. (default:
        ``False``).

    .. versionadded:: 2.5
    """
    name = 'E-mail'
    aliases = ['email', 'eml']
    filenames = ['*.eml']
    mimetypes = ['message/rfc822']

    def __init__(self, **options):
        super().__init__(EmailHeaderLexer, MIMELexer, Comment, **options)