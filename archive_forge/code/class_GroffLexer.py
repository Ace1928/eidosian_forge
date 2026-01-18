import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
class GroffLexer(RegexLexer):
    """
    Lexer for the (g)roff typesetting language, supporting groff
    extensions. Mainly useful for highlighting manpage sources.

    .. versionadded:: 0.6
    """
    name = 'Groff'
    aliases = ['groff', 'nroff', 'man']
    filenames = ['*.[1234567]', '*.man']
    mimetypes = ['application/x-troff', 'text/troff']
    tokens = {'root': [('(\\.)(\\w+)', bygroups(Text, Keyword), 'request'), ('\\.', Punctuation, 'request'), ('[^\\\\\\n]+', Text, 'textline'), default('textline')], 'textline': [include('escapes'), ('[^\\\\\\n]+', Text), ('\\n', Text, '#pop')], 'escapes': [('\\\\"[^\\n]*', Comment), ('\\\\[fn]\\w', String.Escape), ('\\\\\\(.{2}', String.Escape), ('\\\\.\\[.*\\]', String.Escape), ('\\\\.', String.Escape), ('\\\\\\n', Text, 'request')], 'request': [('\\n', Text, '#pop'), include('escapes'), ('"[^\\n"]+"', String.Double), ('\\d+', Number), ('\\S+', String), ('\\s+', Text)]}

    def analyse_text(text):
        if text[:1] != '.':
            return False
        if text[:3] == '.\\"':
            return True
        if text[:4] == '.TH ':
            return True
        if text[1:3].isalnum() and text[3].isspace():
            return 0.9