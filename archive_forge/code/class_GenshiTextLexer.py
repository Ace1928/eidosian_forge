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
class GenshiTextLexer(RegexLexer):
    """
    A lexer that highlights `genshi <http://genshi.edgewall.org/>`_ text
    templates.
    """
    name = 'Genshi Text'
    aliases = ['genshitext']
    mimetypes = ['application/x-genshi-text', 'text/x-genshi']
    tokens = {'root': [('[^#$\\s]+', Other), ('^(\\s*)(##.*)$', bygroups(Text, Comment)), ('^(\\s*)(#)', bygroups(Text, Comment.Preproc), 'directive'), include('variable'), ('[#$\\s]', Other)], 'directive': [('\\n', Text, '#pop'), ('(?:def|for|if)\\s+.*', using(PythonLexer), '#pop'), ('(choose|when|with)([^\\S\\n]+)(.*)', bygroups(Keyword, Text, using(PythonLexer)), '#pop'), ('(choose|otherwise)\\b', Keyword, '#pop'), ('(end\\w*)([^\\S\\n]*)(.*)', bygroups(Keyword, Text, Comment), '#pop')], 'variable': [('(?<!\\$)(\\$\\{)(.+?)(\\})', bygroups(Comment.Preproc, using(PythonLexer), Comment.Preproc)), ('(?<!\\$)(\\$)([a-zA-Z_][\\w.]*)', Name.Variable)]}