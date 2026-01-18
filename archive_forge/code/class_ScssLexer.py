import re
import copy
from pygments.lexer import ExtendedRegexLexer, RegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import iteritems
class ScssLexer(RegexLexer):
    """
    For SCSS stylesheets.
    """
    name = 'SCSS'
    aliases = ['scss']
    filenames = ['*.scss']
    mimetypes = ['text/x-scss']
    flags = re.IGNORECASE | re.DOTALL
    tokens = {'root': [('\\s+', Text), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline), ('@import', Keyword, 'value'), ('@for', Keyword, 'for'), ('@(debug|warn|if|while)', Keyword, 'value'), ('(@mixin)( [\\w-]+)', bygroups(Keyword, Name.Function), 'value'), ('(@include)( [\\w-]+)', bygroups(Keyword, Name.Decorator), 'value'), ('@extend', Keyword, 'selector'), ('(@media)(\\s+)', bygroups(Keyword, Text), 'value'), ('@[\\w-]+', Keyword, 'selector'), ('(\\$[\\w-]*\\w)([ \\t]*:)', bygroups(Name.Variable, Operator), 'value'), default('selector')], 'attr': [('[^\\s:="\\[]+', Name.Attribute), ('#\\{', String.Interpol, 'interpolation'), ('[ \\t]*:', Operator, 'value'), default('#pop')], 'inline-comment': [('(\\\\#|#(?=[^{])|\\*(?=[^/])|[^#*])+', Comment.Multiline), ('#\\{', String.Interpol, 'interpolation'), ('\\*/', Comment, '#pop')]}
    for group, common in iteritems(common_sass_tokens):
        tokens[group] = copy.copy(common)
    tokens['value'].extend([('\\n', Text), ('[;{}]', Punctuation, '#pop')])
    tokens['selector'].extend([('\\n', Text), ('[;{}]', Punctuation, '#pop')])