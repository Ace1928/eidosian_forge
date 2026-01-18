import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
class TexLexer(RegexLexer):
    """
    Lexer for the TeX and LaTeX typesetting languages.
    """
    name = 'TeX'
    aliases = ['tex', 'latex']
    filenames = ['*.tex', '*.aux', '*.toc']
    mimetypes = ['text/x-tex', 'text/x-latex']
    tokens = {'general': [('%.*?\\n', Comment), ('[{}]', Name.Builtin), ('[&_^]', Name.Builtin)], 'root': [('\\\\\\[', String.Backtick, 'displaymath'), ('\\\\\\(', String, 'inlinemath'), ('\\$\\$', String.Backtick, 'displaymath'), ('\\$', String, 'inlinemath'), ('\\\\([a-zA-Z]+|.)', Keyword, 'command'), ('\\\\$', Keyword), include('general'), ('[^\\\\$%&_^{}]+', Text)], 'math': [('\\\\([a-zA-Z]+|.)', Name.Variable), include('general'), ('[0-9]+', Number), ('[-=!+*/()\\[\\]]', Operator), ('[^=!+*/()\\[\\]\\\\$%&_^{}0-9-]+', Name.Builtin)], 'inlinemath': [('\\\\\\)', String, '#pop'), ('\\$', String, '#pop'), include('math')], 'displaymath': [('\\\\\\]', String, '#pop'), ('\\$\\$', String, '#pop'), ('\\$', Name.Builtin), include('math')], 'command': [('\\[.*?\\]', Name.Attribute), ('\\*', Keyword), default('#pop')]}

    def analyse_text(text):
        for start in ('\\documentclass', '\\input', '\\documentstyle', '\\relax'):
            if text[:len(start)] == start:
                return True