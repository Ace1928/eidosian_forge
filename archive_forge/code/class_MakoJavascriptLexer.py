import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer, LassoLexer
from pygments.lexers.css import CssLexer
from pygments.lexers.php import PhpLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.perl import PerlLexer
from pygments.lexers.jvm import JavaLexer, TeaLangLexer
from pygments.lexers.data import YamlLexer
from pygments.lexer import Lexer, DelegatingLexer, RegexLexer, bygroups, \
from pygments.token import Error, Punctuation, Whitespace, \
from pygments.util import html_doctype_matches, looks_like_xml
class MakoJavascriptLexer(DelegatingLexer):
    """
    Subclass of the `MakoLexer` that highlights unlexed data
    with the `JavascriptLexer`.

    .. versionadded:: 0.7
    """
    name = 'JavaScript+Mako'
    aliases = ['js+mako', 'javascript+mako']
    mimetypes = ['application/x-javascript+mako', 'text/x-javascript+mako', 'text/javascript+mako']

    def __init__(self, **options):
        super(MakoJavascriptLexer, self).__init__(JavascriptLexer, MakoLexer, **options)