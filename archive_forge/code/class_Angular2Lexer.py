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
class Angular2Lexer(RegexLexer):
    """
    Generic
    `angular2 <http://victorsavkin.com/post/119943127151/angular-2-template-syntax>`_
    template lexer.

    Highlights only the Angular template tags (stuff between `{{` and `}}` and
    special attributes: '(event)=', '[property]=', '[(twoWayBinding)]=').
    Everything else is left for a delegating lexer.

    .. versionadded:: 2.1
    """
    name = 'Angular2'
    aliases = ['ng2']
    tokens = {'root': [('[^{([*#]+', Other), ('(\\{\\{)(\\s*)', bygroups(Comment.Preproc, Text), 'ngExpression'), ('([([]+)([\\w:.-]+)([\\])]+)(\\s*)(=)(\\s*)', bygroups(Punctuation, Name.Attribute, Punctuation, Text, Operator, Text), 'attr'), ('([([]+)([\\w:.-]+)([\\])]+)(\\s*)', bygroups(Punctuation, Name.Attribute, Punctuation, Text)), ('([*#])([\\w:.-]+)(\\s*)(=)(\\s*)', bygroups(Punctuation, Name.Attribute, Punctuation, Operator), 'attr'), ('([*#])([\\w:.-]+)(\\s*)', bygroups(Punctuation, Name.Attribute, Punctuation))], 'ngExpression': [('\\s+(\\|\\s+)?', Text), ('\\}\\}', Comment.Preproc, '#pop'), (':?(true|false)', String.Boolean), (':?"(\\\\\\\\|\\\\"|[^"])*"', String.Double), (":?'(\\\\\\\\|\\\\'|[^'])*'", String.Single), ('[0-9](\\.[0-9]*)?(eE[+-][0-9])?[flFLdD]?|0[xX][0-9a-fA-F]+[Ll]?', Number), ('[a-zA-Z][\\w-]*(\\(.*\\))?', Name.Variable), ('\\.[\\w-]+(\\(.*\\))?', Name.Variable), ('(\\?)(\\s*)([^}\\s]+)(\\s*)(:)(\\s*)([^}\\s]+)(\\s*)', bygroups(Operator, Text, String, Text, Operator, Text, String, Text))], 'attr': [('".*?"', String, '#pop'), ("'.*?'", String, '#pop'), ('[^\\s>]+', String, '#pop')]}