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
class HtmlGenshiLexer(DelegatingLexer):
    """
    A lexer that highlights `genshi <http://genshi.edgewall.org/>`_ and
    `kid <http://kid-templating.org/>`_ kid HTML templates.
    """
    name = 'HTML+Genshi'
    aliases = ['html+genshi', 'html+kid']
    alias_filenames = ['*.html', '*.htm', '*.xhtml']
    mimetypes = ['text/html+genshi']

    def __init__(self, **options):
        super(HtmlGenshiLexer, self).__init__(HtmlLexer, GenshiMarkupLexer, **options)

    def analyse_text(text):
        rv = 0.0
        if re.search('\\$\\{.*?\\}', text) is not None:
            rv += 0.2
        if re.search('py:(.*?)=["\']', text) is not None:
            rv += 0.2
        return rv + HtmlLexer.analyse_text(text) - 0.01