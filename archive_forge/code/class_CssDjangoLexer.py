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
class CssDjangoLexer(DelegatingLexer):
    """
    Subclass of the `DjangoLexer` that highlights unlexed data with the
    `CssLexer`.
    """
    name = 'CSS+Django/Jinja'
    aliases = ['css+django', 'css+jinja']
    alias_filenames = ['*.css']
    mimetypes = ['text/css+django', 'text/css+jinja']

    def __init__(self, **options):
        super(CssDjangoLexer, self).__init__(CssLexer, DjangoLexer, **options)

    def analyse_text(text):
        return DjangoLexer.analyse_text(text) - 0.05