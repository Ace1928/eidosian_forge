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
class GenshiMarkupLexer(RegexLexer):
    """
    Base lexer for Genshi markup, used by `HtmlGenshiLexer` and
    `GenshiLexer`.
    """
    flags = re.DOTALL
    tokens = {'root': [('[^<$]+', Other), ('(<\\?python)(.*?)(\\?>)', bygroups(Comment.Preproc, using(PythonLexer), Comment.Preproc)), ('<\\s*(script|style)\\s*.*?>.*?<\\s*/\\1\\s*>', Other), ('<\\s*py:[a-zA-Z0-9]+', Name.Tag, 'pytag'), ('<\\s*[a-zA-Z0-9:.]+', Name.Tag, 'tag'), include('variable'), ('[<$]', Other)], 'pytag': [('\\s+', Text), ('[\\w:-]+\\s*=', Name.Attribute, 'pyattr'), ('/?\\s*>', Name.Tag, '#pop')], 'pyattr': [('(")(.*?)(")', bygroups(String, using(PythonLexer), String), '#pop'), ("(')(.*?)(')", bygroups(String, using(PythonLexer), String), '#pop'), ('[^\\s>]+', String, '#pop')], 'tag': [('\\s+', Text), ('py:[\\w-]+\\s*=', Name.Attribute, 'pyattr'), ('[\\w:-]+\\s*=', Name.Attribute, 'attr'), ('/?\\s*>', Name.Tag, '#pop')], 'attr': [('"', String, 'attr-dstring'), ("'", String, 'attr-sstring'), ('[^\\s>]*', String, '#pop')], 'attr-dstring': [('"', String, '#pop'), include('strings'), ("'", String)], 'attr-sstring': [("'", String, '#pop'), include('strings'), ("'", String)], 'strings': [('[^"\'$]+', String), include('variable')], 'variable': [('(?<!\\$)(\\$\\{)(.+?)(\\})', bygroups(Comment.Preproc, using(PythonLexer), Comment.Preproc)), ('(?<!\\$)(\\$)([a-zA-Z_][\\w\\.]*)', Name.Variable)]}