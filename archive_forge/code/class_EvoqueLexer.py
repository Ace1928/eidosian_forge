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
class EvoqueLexer(RegexLexer):
    """
    For files using the Evoque templating system.

    .. versionadded:: 1.1
    """
    name = 'Evoque'
    aliases = ['evoque']
    filenames = ['*.evoque']
    mimetypes = ['application/x-evoque']
    flags = re.DOTALL
    tokens = {'root': [('[^#$]+', Other), ('#\\[', Comment.Multiline, 'comment'), ('\\$\\$', Other), ('\\$\\w+:[^$\\n]*\\$', Comment.Multiline), ('(\\$)(begin|end)(\\{(%)?)(.*?)((?(4)%)\\})', bygroups(Punctuation, Name.Builtin, Punctuation, None, String, Punctuation)), ('(\\$)(evoque|overlay)(\\{(%)?)(\\s*[#\\w\\-"\\\'.]+[^=,%}]+?)?(.*?)((?(4)%)\\})', bygroups(Punctuation, Name.Builtin, Punctuation, None, String, using(PythonLexer), Punctuation)), ('(\\$)(\\w+)(\\{(%)?)(.*?)((?(4)%)\\})', bygroups(Punctuation, Name.Builtin, Punctuation, None, using(PythonLexer), Punctuation)), ('(\\$)(else|rof|fi)', bygroups(Punctuation, Name.Builtin)), ('(\\$\\{(%)?)(.*?)((!)(.*?))?((?(2)%)\\})', bygroups(Punctuation, None, using(PythonLexer), Name.Builtin, None, None, Punctuation)), ('#', Other)], 'comment': [('[^\\]#]', Comment.Multiline), ('#\\[', Comment.Multiline, '#push'), ('\\]#', Comment.Multiline, '#pop'), ('[\\]#]', Comment.Multiline)]}