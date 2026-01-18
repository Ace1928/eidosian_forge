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
class CheetahHtmlLexer(DelegatingLexer):
    """
    Subclass of the `CheetahLexer` that highlights unlexed data
    with the `HtmlLexer`.
    """
    name = 'HTML+Cheetah'
    aliases = ['html+cheetah', 'html+spitfire', 'htmlcheetah']
    mimetypes = ['text/html+cheetah', 'text/html+spitfire']

    def __init__(self, **options):
        super(CheetahHtmlLexer, self).__init__(HtmlLexer, CheetahLexer, **options)