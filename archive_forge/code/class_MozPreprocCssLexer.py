import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
class MozPreprocCssLexer(DelegatingLexer):
    """
    Subclass of the `MozPreprocHashLexer` that highlights unlexed data with the
    `CssLexer`.

    .. versionadded:: 2.0
    """
    name = 'CSS+mozpreproc'
    aliases = ['css+mozpreproc']
    filenames = ['*.css.in']
    mimetypes = []

    def __init__(self, **options):
        super(MozPreprocCssLexer, self).__init__(CssLexer, MozPreprocPercentLexer, **options)