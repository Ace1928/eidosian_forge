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
class VelocityHtmlLexer(DelegatingLexer):
    """
    Subclass of the `VelocityLexer` that highlights unlexed data
    with the `HtmlLexer`.

    """
    name = 'HTML+Velocity'
    aliases = ['html+velocity']
    alias_filenames = ['*.html', '*.fhtml']
    mimetypes = ['text/html+velocity']

    def __init__(self, **options):
        super(VelocityHtmlLexer, self).__init__(HtmlLexer, VelocityLexer, **options)