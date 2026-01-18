import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import looks_like_xml, html_doctype_matches
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.jvm import ScalaLexer
from pygments.lexers.css import CssLexer, _indentation, _starts_block
from pygments.lexers.ruby import RubyLexer
class HamlLexer(ExtendedRegexLexer):
    """
    For Haml markup.

    .. versionadded:: 1.3
    """
    name = 'Haml'
    aliases = ['haml']
    filenames = ['*.haml']
    mimetypes = ['text/x-haml']
    flags = re.IGNORECASE
    _dot = '(?: \\|\\n(?=.* \\|)|.)'
    _comma_dot = '(?:,\\s*\\n|' + _dot + ')'
    tokens = {'root': [('[ \\t]*\\n', Text), ('[ \\t]*', _indentation)], 'css': [('\\.[\\w:-]+', Name.Class, 'tag'), ('\\#[\\w:-]+', Name.Function, 'tag')], 'eval-or-plain': [('[&!]?==', Punctuation, 'plain'), ('([&!]?[=~])(' + _comma_dot + '*\\n)', bygroups(Punctuation, using(RubyLexer)), 'root'), default('plain')], 'content': [include('css'), ('%[\\w:-]+', Name.Tag, 'tag'), ('!!!' + _dot + '*\\n', Name.Namespace, '#pop'), ('(/)(\\[' + _dot + '*?\\])(' + _dot + '*\\n)', bygroups(Comment, Comment.Special, Comment), '#pop'), ('/' + _dot + '*\\n', _starts_block(Comment, 'html-comment-block'), '#pop'), ('-#' + _dot + '*\\n', _starts_block(Comment.Preproc, 'haml-comment-block'), '#pop'), ('(-)(' + _comma_dot + '*\\n)', bygroups(Punctuation, using(RubyLexer)), '#pop'), (':' + _dot + '*\\n', _starts_block(Name.Decorator, 'filter-block'), '#pop'), include('eval-or-plain')], 'tag': [include('css'), ('\\{(,\\n|' + _dot + ')*?\\}', using(RubyLexer)), ('\\[' + _dot + '*?\\]', using(RubyLexer)), ('\\(', Text, 'html-attributes'), ('/[ \\t]*\\n', Punctuation, '#pop:2'), ('[<>]{1,2}(?=[ \\t=])', Punctuation), include('eval-or-plain')], 'plain': [('([^#\\n]|#[^{\\n]|(\\\\\\\\)*\\\\#\\{)+', Text), ('(#\\{)(' + _dot + '*?)(\\})', bygroups(String.Interpol, using(RubyLexer), String.Interpol)), ('\\n', Text, 'root')], 'html-attributes': [('\\s+', Text), ('[\\w:-]+[ \\t]*=', Name.Attribute, 'html-attribute-value'), ('[\\w:-]+', Name.Attribute), ('\\)', Text, '#pop')], 'html-attribute-value': [('[ \\t]+', Text), ('\\w+', Name.Variable, '#pop'), ('@\\w+', Name.Variable.Instance, '#pop'), ('\\$\\w+', Name.Variable.Global, '#pop'), ("'(\\\\\\\\|\\\\'|[^'\\n])*'", String, '#pop'), ('"(\\\\\\\\|\\\\"|[^"\\n])*"', String, '#pop')], 'html-comment-block': [(_dot + '+', Comment), ('\\n', Text, 'root')], 'haml-comment-block': [(_dot + '+', Comment.Preproc), ('\\n', Text, 'root')], 'filter-block': [('([^#\\n]|#[^{\\n]|(\\\\\\\\)*\\\\#\\{)+', Name.Decorator), ('(#\\{)(' + _dot + '*?)(\\})', bygroups(String.Interpol, using(RubyLexer), String.Interpol)), ('\\n', Text, 'root')]}