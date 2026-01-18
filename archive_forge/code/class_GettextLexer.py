import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import ClassNotFound
class GettextLexer(RegexLexer):
    """
    Lexer for Gettext catalog files.

    .. versionadded:: 0.9
    """
    name = 'Gettext Catalog'
    aliases = ['pot', 'po']
    filenames = ['*.pot', '*.po']
    mimetypes = ['application/x-gettext', 'text/x-gettext', 'text/gettext']
    tokens = {'root': [('^#,\\s.*?$', Keyword.Type), ('^#:\\s.*?$', Keyword.Declaration), ('^(#|#\\.\\s|#\\|\\s|#~\\s|#\\s).*$', Comment.Single), ('^(")([A-Za-z-]+:)(.*")$', bygroups(String, Name.Property, String)), ('^".*"$', String), ('^(msgid|msgid_plural|msgstr|msgctxt)(\\s+)(".*")$', bygroups(Name.Variable, Text, String)), ('^(msgstr\\[)(\\d)(\\])(\\s+)(".*")$', bygroups(Name.Variable, Number.Integer, Name.Variable, Text, String))]}