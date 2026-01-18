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
class VelocityXmlLexer(DelegatingLexer):
    """
    Subclass of the `VelocityLexer` that highlights unlexed data
    with the `XmlLexer`.

    """
    name = 'XML+Velocity'
    aliases = ['xml+velocity']
    alias_filenames = ['*.xml', '*.vm']
    mimetypes = ['application/xml+velocity']

    def __init__(self, **options):
        super(VelocityXmlLexer, self).__init__(XmlLexer, VelocityLexer, **options)

    def analyse_text(text):
        rv = VelocityLexer.analyse_text(text) - 0.01
        if looks_like_xml(text):
            rv += 0.4
        return rv