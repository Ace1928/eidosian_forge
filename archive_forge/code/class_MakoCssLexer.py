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
class MakoCssLexer(DelegatingLexer):
    """
    Subclass of the `MakoLexer` that highlights unlexed data
    with the `CssLexer`.

    .. versionadded:: 0.7
    """
    name = 'CSS+Mako'
    aliases = ['css+mako']
    mimetypes = ['text/css+mako']

    def __init__(self, **options):
        super(MakoCssLexer, self).__init__(CssLexer, MakoLexer, **options)