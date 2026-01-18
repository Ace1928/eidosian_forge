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
class SspLexer(DelegatingLexer):
    """
    Lexer for Scalate Server Pages.

    .. versionadded:: 1.4
    """
    name = 'Scalate Server Page'
    aliases = ['ssp']
    filenames = ['*.ssp']
    mimetypes = ['application/x-ssp']

    def __init__(self, **options):
        super(SspLexer, self).__init__(XmlLexer, JspRootLexer, **options)

    def analyse_text(text):
        rv = 0.0
        if re.search('val \\w+\\s*:', text):
            rv += 0.6
        if looks_like_xml(text):
            rv += 0.2
        if '<%' in text and '%>' in text:
            rv += 0.1
        return rv