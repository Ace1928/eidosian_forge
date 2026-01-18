import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
class MozPreprocPercentLexer(MozPreprocHashLexer):
    """
    Lexer for Mozilla Preprocessor files (with '%' as the marker).

    Other data is left untouched.

    .. versionadded:: 2.0
    """
    name = 'mozpercentpreproc'
    aliases = [name]
    filenames = []
    mimetypes = []
    tokens = {'root': [('^%', Comment.Preproc, ('expr', 'exprstart')), ('.+', Other)]}