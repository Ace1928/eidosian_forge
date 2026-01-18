import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import looks_like_xml, html_doctype_matches
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.jvm import ScalaLexer
from pygments.lexers.css import CssLexer, _indentation, _starts_block
from pygments.lexers.ruby import RubyLexer
class ScamlLexer(ExtendedRegexLexer):
    """
    For `Scaml markup <http://scalate.fusesource.org/>`_.  Scaml is Haml for Scala.

    .. versionadded:: 1.4
    """
    name = 'Scaml'
    aliases = ['scaml']
    filenames = ['*.scaml']
    mimetypes = ['text/x-scaml']
    flags = re.IGNORECASE
    _dot = '.'
    tokens = {'root': [('[ \\t]*\\n', Text), ('[ \\t]*', _indentation)], 'css': [('\\.[\\w:-]+', Name.Class, 'tag'), ('\\#[\\w:-]+', Name.Function, 'tag')], 'eval-or-plain': [('[&!]?==', Punctuation, 'plain'), ('([&!]?[=~])(' + _dot + '*\\n)', bygroups(Punctuation, using(ScalaLexer)), 'root'), default('plain')], 'content': [include('css'), ('%[\\w:-]+', Name.Tag, 'tag'), ('!!!' + _dot + '*\\n', Name.Namespace, '#pop'), ('(/)(\\[' + _dot + '*?\\])(' + _dot + '*\\n)', bygroups(Comment, Comment.Special, Comment), '#pop'), ('/' + _dot + '*\\n', _starts_block(Comment, 'html-comment-block'), '#pop'), ('-#' + _dot + '*\\n', _starts_block(Comment.Preproc, 'scaml-comment-block'), '#pop'), ('(-@\\s*)(import)?(' + _dot + '*\\n)', bygroups(Punctuation, Keyword, using(ScalaLexer)), '#pop'), ('(-)(' + _dot + '*\\n)', bygroups(Punctuation, using(ScalaLexer)), '#pop'), (':' + _dot + '*\\n', _starts_block(Name.Decorator, 'filter-block'), '#pop'), include('eval-or-plain')], 'tag': [include('css'), ('\\{(,\\n|' + _dot + ')*?\\}', using(ScalaLexer)), ('\\[' + _dot + '*?\\]', using(ScalaLexer)), ('\\(', Text, 'html-attributes'), ('/[ \\t]*\\n', Punctuation, '#pop:2'), ('[<>]{1,2}(?=[ \\t=])', Punctuation), include('eval-or-plain')], 'plain': [('([^#\\n]|#[^{\\n]|(\\\\\\\\)*\\\\#\\{)+', Text), ('(#\\{)(' + _dot + '*?)(\\})', bygroups(String.Interpol, using(ScalaLexer), String.Interpol)), ('\\n', Text, 'root')], 'html-attributes': [('\\s+', Text), ('[\\w:-]+[ \\t]*=', Name.Attribute, 'html-attribute-value'), ('[\\w:-]+', Name.Attribute), ('\\)', Text, '#pop')], 'html-attribute-value': [('[ \\t]+', Text), ('\\w+', Name.Variable, '#pop'), ('@\\w+', Name.Variable.Instance, '#pop'), ('\\$\\w+', Name.Variable.Global, '#pop'), ("'(\\\\\\\\|\\\\'|[^'\\n])*'", String, '#pop'), ('"(\\\\\\\\|\\\\"|[^"\\n])*"', String, '#pop')], 'html-comment-block': [(_dot + '+', Comment), ('\\n', Text, 'root')], 'scaml-comment-block': [(_dot + '+', Comment.Preproc), ('\\n', Text, 'root')], 'filter-block': [('([^#\\n]|#[^{\\n]|(\\\\\\\\)*\\\\#\\{)+', Name.Decorator), ('(#\\{)(' + _dot + '*?)(\\})', bygroups(String.Interpol, using(ScalaLexer), String.Interpol)), ('\\n', Text, 'root')]}