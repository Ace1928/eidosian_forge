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
class JavascriptErbLexer(DelegatingLexer):
    """
    Subclass of `ErbLexer` which highlights unlexed data with the
    `JavascriptLexer`.
    """
    name = 'JavaScript+Ruby'
    aliases = ['js+erb', 'javascript+erb', 'js+ruby', 'javascript+ruby']
    alias_filenames = ['*.js']
    mimetypes = ['application/x-javascript+ruby', 'text/x-javascript+ruby', 'text/javascript+ruby']

    def __init__(self, **options):
        super(JavascriptErbLexer, self).__init__(JavascriptLexer, ErbLexer, **options)

    def analyse_text(text):
        return ErbLexer.analyse_text(text) - 0.05