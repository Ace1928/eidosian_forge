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
class VelocityLexer(RegexLexer):
    """
    Generic `Velocity <http://velocity.apache.org/>`_ template lexer.

    Just highlights velocity directives and variable references, other
    data is left untouched by the lexer.
    """
    name = 'Velocity'
    aliases = ['velocity']
    filenames = ['*.vm', '*.fhtml']
    flags = re.MULTILINE | re.DOTALL
    identifier = '[a-zA-Z_]\\w*'
    tokens = {'root': [('[^{#$]+', Other), ('(#)(\\*.*?\\*)(#)', bygroups(Comment.Preproc, Comment, Comment.Preproc)), ('(##)(.*?$)', bygroups(Comment.Preproc, Comment)), ('(#\\{?)(' + identifier + ')(\\}?)(\\s?\\()', bygroups(Comment.Preproc, Name.Function, Comment.Preproc, Punctuation), 'directiveparams'), ('(#\\{?)(' + identifier + ')(\\}|\\b)', bygroups(Comment.Preproc, Name.Function, Comment.Preproc)), ('\\$\\{?', Punctuation, 'variable')], 'variable': [(identifier, Name.Variable), ('\\(', Punctuation, 'funcparams'), ('(\\.)(' + identifier + ')', bygroups(Punctuation, Name.Variable), '#push'), ('\\}', Punctuation, '#pop'), default('#pop')], 'directiveparams': [('(&&|\\|\\||==?|!=?|[-<>+*%&|^/])|\\b(eq|ne|gt|lt|ge|le|not|in)\\b', Operator), ('\\[', Operator, 'rangeoperator'), ('\\b' + identifier + '\\b', Name.Function), include('funcparams')], 'rangeoperator': [('\\.\\.', Operator), include('funcparams'), ('\\]', Operator, '#pop')], 'funcparams': [('\\$\\{?', Punctuation, 'variable'), ('\\s+', Text), ('[,:]', Punctuation), ('"(\\\\\\\\|\\\\"|[^"])*"', String.Double), ("'(\\\\\\\\|\\\\'|[^'])*'", String.Single), ('0[xX][0-9a-fA-F]+[Ll]?', Number), ('\\b[0-9]+\\b', Number), ('(true|false|null)\\b', Keyword.Constant), ('\\(', Punctuation, '#push'), ('\\)', Punctuation, '#pop'), ('\\{', Punctuation, '#push'), ('\\}', Punctuation, '#pop'), ('\\[', Punctuation, '#push'), ('\\]', Punctuation, '#pop')]}

    def analyse_text(text):
        rv = 0.0
        if re.search('#\\{?macro\\}?\\(.*?\\).*?#\\{?end\\}?', text):
            rv += 0.25
        if re.search('#\\{?if\\}?\\(.+?\\).*?#\\{?end\\}?', text):
            rv += 0.15
        if re.search('#\\{?foreach\\}?\\(.+?\\).*?#\\{?end\\}?', text):
            rv += 0.15
        if re.search('\\$\\{?[a-zA-Z_]\\w*(\\([^)]*\\))?(\\.\\w+(\\([^)]*\\))?)*\\}?', text):
            rv += 0.01
        return rv