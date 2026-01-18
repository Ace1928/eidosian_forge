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
class HandlebarsLexer(RegexLexer):
    """
    Generic `handlebars <http://handlebarsjs.com/>` template lexer.

    Highlights only the Handlebars template tags (stuff between `{{` and `}}`).
    Everything else is left for a delegating lexer.

    .. versionadded:: 2.0
    """
    name = 'Handlebars'
    aliases = ['handlebars']
    tokens = {'root': [('[^{]+', Other), ('\\{\\{!.*\\}\\}', Comment), ('(\\{\\{\\{)(\\s*)', bygroups(Comment.Special, Text), 'tag'), ('(\\{\\{)(\\s*)', bygroups(Comment.Preproc, Text), 'tag')], 'tag': [('\\s+', Text), ('\\}\\}\\}', Comment.Special, '#pop'), ('\\}\\}', Comment.Preproc, '#pop'), ('([#/]*)(each|if|unless|else|with|log|in(line)?)', bygroups(Keyword, Keyword)), ('#\\*inline', Keyword), ('([#/])([\\w-]+)', bygroups(Name.Function, Name.Function)), ('([\\w-]+)(=)', bygroups(Name.Attribute, Operator)), ('(>)(\\s*)(@partial-block)', bygroups(Keyword, Text, Keyword)), ('(#?>)(\\s*)([\\w-]+)', bygroups(Keyword, Text, Name.Variable)), ('(>)(\\s*)(\\()', bygroups(Keyword, Text, Punctuation), 'dynamic-partial'), include('generic')], 'dynamic-partial': [('\\s+', Text), ('\\)', Punctuation, '#pop'), ('(lookup)(\\s+)(\\.|this)(\\s+)', bygroups(Keyword, Text, Name.Variable, Text)), ('(lookup)(\\s+)(\\S+)', bygroups(Keyword, Text, using(this, state='variable'))), ('[\\w-]+', Name.Function), include('generic')], 'variable': [('[a-zA-Z][\\w-]*', Name.Variable), ('\\.[\\w-]+', Name.Variable), ('(this\\/|\\.\\/|(\\.\\.\\/)+)[\\w-]+', Name.Variable)], 'generic': [include('variable'), (':?"(\\\\\\\\|\\\\"|[^"])*"', String.Double), (":?'(\\\\\\\\|\\\\'|[^'])*'", String.Single), ('[0-9](\\.[0-9]*)?(eE[+-][0-9])?[flFLdD]?|0[xX][0-9a-fA-F]+[Ll]?', Number)]}