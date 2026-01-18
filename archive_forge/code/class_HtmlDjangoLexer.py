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
class HtmlDjangoLexer(DelegatingLexer):
    """
    Subclass of the `DjangoLexer` that highlights unlexed data with the
    `HtmlLexer`.

    Nested Javascript and CSS is highlighted too.
    """
    name = 'HTML+Django/Jinja'
    aliases = ['html+django', 'html+jinja', 'htmldjango']
    alias_filenames = ['*.html', '*.htm', '*.xhtml']
    mimetypes = ['text/html+django', 'text/html+jinja']

    def __init__(self, **options):
        super(HtmlDjangoLexer, self).__init__(HtmlLexer, DjangoLexer, **options)

    def analyse_text(text):
        rv = DjangoLexer.analyse_text(text) - 0.01
        if html_doctype_matches(text):
            rv += 0.5
        return rv