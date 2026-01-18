import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
class BBCodeLexer(RegexLexer):
    """
    A lexer that highlights BBCode(-like) syntax.

    .. versionadded:: 0.6
    """
    name = 'BBCode'
    aliases = ['bbcode']
    mimetypes = ['text/x-bbcode']
    tokens = {'root': [('[^[]+', Text), ('\\[/?\\w+', Keyword, 'tag'), ('\\[', Text)], 'tag': [('\\s+', Text), ('(\\w+)(=)("?[^\\s"\\]]+"?)', bygroups(Name.Attribute, Operator, String)), ('(=)("?[^\\s"\\]]+"?)', bygroups(Operator, String)), ('\\]', Keyword, '#pop')]}