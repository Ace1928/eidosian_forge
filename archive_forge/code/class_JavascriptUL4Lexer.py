import re
from pygments.lexer import RegexLexer, DelegatingLexer, bygroups, words, include
from pygments.token import Comment, Text, Keyword, String, Number, Literal, \
from pygments.lexers.web import HtmlLexer, XmlLexer, CssLexer, JavascriptLexer
from pygments.lexers.python import PythonLexer
class JavascriptUL4Lexer(DelegatingLexer):
    """
    Lexer for UL4 embedded in Javascript.
    """
    name = 'Javascript+UL4'
    aliases = ['js+ul4']
    filenames = ['*.jsul4']

    def __init__(self, **options):
        super().__init__(JavascriptLexer, UL4Lexer, **options)