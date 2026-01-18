import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
class MozPreprocJavascriptLexer(DelegatingLexer):
    """
    Subclass of the `MozPreprocHashLexer` that highlights unlexed data with the
    `JavascriptLexer`.

    .. versionadded:: 2.0
    """
    name = 'Javascript+mozpreproc'
    aliases = ['javascript+mozpreproc']
    filenames = ['*.js.in']
    mimetypes = []

    def __init__(self, **options):
        super(MozPreprocJavascriptLexer, self).__init__(JavascriptLexer, MozPreprocHashLexer, **options)