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
class TwigLexer(RegexLexer):
    """
    `Twig <http://twig.sensiolabs.org/>`_ template lexer.

    It just highlights Twig code between the preprocessor directives,
    other data is left untouched by the lexer.

    .. versionadded:: 2.0
    """
    name = 'Twig'
    aliases = ['twig']
    mimetypes = ['application/x-twig']
    flags = re.M | re.S
    _ident_char = '[\\\\\\w-]|[^\\x00-\\x7f]'
    _ident_begin = '(?:[\\\\_a-z]|[^\\x00-\\x7f])'
    _ident_end = '(?:' + _ident_char + ')*'
    _ident_inner = _ident_begin + _ident_end
    tokens = {'root': [('[^{]+', Other), ('\\{\\{', Comment.Preproc, 'var'), ('\\{\\#.*?\\#\\}', Comment), ('(\\{%)(-?\\s*)(raw)(\\s*-?)(%\\})(.*?)(\\{%)(-?\\s*)(endraw)(\\s*-?)(%\\})', bygroups(Comment.Preproc, Text, Keyword, Text, Comment.Preproc, Other, Comment.Preproc, Text, Keyword, Text, Comment.Preproc)), ('(\\{%)(-?\\s*)(verbatim)(\\s*-?)(%\\})(.*?)(\\{%)(-?\\s*)(endverbatim)(\\s*-?)(%\\})', bygroups(Comment.Preproc, Text, Keyword, Text, Comment.Preproc, Other, Comment.Preproc, Text, Keyword, Text, Comment.Preproc)), ('(\\{%%)(-?\\s*)(filter)(\\s+)(%s)' % _ident_inner, bygroups(Comment.Preproc, Text, Keyword, Text, Name.Function), 'tag'), ('(\\{%)(-?\\s*)([a-zA-Z_]\\w*)', bygroups(Comment.Preproc, Text, Keyword), 'tag'), ('\\{', Other)], 'varnames': [('(\\|)(\\s*)(%s)' % _ident_inner, bygroups(Operator, Text, Name.Function)), ('(is)(\\s+)(not)?(\\s*)(%s)' % _ident_inner, bygroups(Keyword, Text, Keyword, Text, Name.Function)), ('(?i)(true|false|none|null)\\b', Keyword.Pseudo), ('(in|not|and|b-and|or|b-or|b-xor|isif|elseif|else|importconstant|defined|divisibleby|empty|even|iterable|odd|sameasmatches|starts\\s+with|ends\\s+with)\\b', Keyword), ('(loop|block|parent)\\b', Name.Builtin), (_ident_inner, Name.Variable), ('\\.' + _ident_inner, Name.Variable), ('\\.[0-9]+', Number), (':?"(\\\\\\\\|\\\\"|[^"])*"', String.Double), (":?'(\\\\\\\\|\\\\'|[^'])*'", String.Single), ('([{}()\\[\\]+\\-*/,:~%]|\\.\\.|\\?|:|\\*\\*|\\/\\/|!=|[><=]=?)', Operator), ('[0-9](\\.[0-9]*)?(eE[+-][0-9])?[flFLdD]?|0[xX][0-9a-fA-F]+[Ll]?', Number)], 'var': [('\\s+', Text), ('(-?)(\\}\\})', bygroups(Text, Comment.Preproc), '#pop'), include('varnames')], 'tag': [('\\s+', Text), ('(-?)(%\\})', bygroups(Text, Comment.Preproc), '#pop'), include('varnames'), ('.', Punctuation)]}