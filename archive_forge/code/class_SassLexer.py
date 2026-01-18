import re
import copy
from pygments.lexer import ExtendedRegexLexer, RegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import iteritems
class SassLexer(ExtendedRegexLexer):
    """
    For Sass stylesheets.

    .. versionadded:: 1.3
    """
    name = 'Sass'
    aliases = ['sass']
    filenames = ['*.sass']
    mimetypes = ['text/x-sass']
    flags = re.IGNORECASE | re.MULTILINE
    tokens = {'root': [('[ \\t]*\\n', Text), ('[ \\t]*', _indentation)], 'content': [('//[^\\n]*', _starts_block(Comment.Single, 'single-comment'), 'root'), ('/\\*[^\\n]*', _starts_block(Comment.Multiline, 'multi-comment'), 'root'), ('@import', Keyword, 'import'), ('@for', Keyword, 'for'), ('@(debug|warn|if|while)', Keyword, 'value'), ('(@mixin)( [\\w-]+)', bygroups(Keyword, Name.Function), 'value'), ('(@include)( [\\w-]+)', bygroups(Keyword, Name.Decorator), 'value'), ('@extend', Keyword, 'selector'), ('@[\\w-]+', Keyword, 'selector'), ('=[\\w-]+', Name.Function, 'value'), ('\\+[\\w-]+', Name.Decorator, 'value'), ('([!$][\\w-]\\w*)([ \\t]*(?:(?:\\|\\|)?=|:))', bygroups(Name.Variable, Operator), 'value'), (':', Name.Attribute, 'old-style-attr'), ('(?=.+?[=:]([^a-z]|$))', Name.Attribute, 'new-style-attr'), default('selector')], 'single-comment': [('.+', Comment.Single), ('\\n', Text, 'root')], 'multi-comment': [('.+', Comment.Multiline), ('\\n', Text, 'root')], 'import': [('[ \\t]+', Text), ('\\S+', String), ('\\n', Text, 'root')], 'old-style-attr': [('[^\\s:="\\[]+', Name.Attribute), ('#\\{', String.Interpol, 'interpolation'), ('[ \\t]*=', Operator, 'value'), default('value')], 'new-style-attr': [('[^\\s:="\\[]+', Name.Attribute), ('#\\{', String.Interpol, 'interpolation'), ('[ \\t]*[=:]', Operator, 'value')], 'inline-comment': [('(\\\\#|#(?=[^\\n{])|\\*(?=[^\\n/])|[^\\n#*])+', Comment.Multiline), ('#\\{', String.Interpol, 'interpolation'), ('\\*/', Comment, '#pop')]}
    for group, common in iteritems(common_sass_tokens):
        tokens[group] = copy.copy(common)
    tokens['value'].append(('\\n', Text, 'root'))
    tokens['selector'].append(('\\n', Text, 'root'))