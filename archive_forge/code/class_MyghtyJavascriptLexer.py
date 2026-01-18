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
class MyghtyJavascriptLexer(DelegatingLexer):
    """
    Subclass of the `MyghtyLexer` that highlights unlexed data
    with the `JavascriptLexer`.

    .. versionadded:: 0.6
    """
    name = 'JavaScript+Myghty'
    aliases = ['js+myghty', 'javascript+myghty']
    mimetypes = ['application/x-javascript+myghty', 'text/x-javascript+myghty', 'text/javascript+mygthy']

    def __init__(self, **options):
        super(MyghtyJavascriptLexer, self).__init__(JavascriptLexer, MyghtyLexer, **options)