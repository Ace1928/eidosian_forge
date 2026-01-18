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
class TeaTemplateRootLexer(RegexLexer):
    """
    Base for the `TeaTemplateLexer`. Yields `Token.Other` for area outside of
    code blocks.

    .. versionadded:: 1.5
    """
    tokens = {'root': [('<%\\S?', Keyword, 'sec'), ('[^<]+', Other), ('<', Other)], 'sec': [('%>', Keyword, '#pop'), ('[\\w\\W]+?(?=%>|\\Z)', using(TeaLangLexer))]}