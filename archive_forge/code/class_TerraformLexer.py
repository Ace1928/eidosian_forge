import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class TerraformLexer(RegexLexer):
    """
    Lexer for `terraformi .tf files <https://www.terraform.io/>`_.

    .. versionadded:: 2.1
    """
    name = 'Terraform'
    aliases = ['terraform', 'tf']
    filenames = ['*.tf']
    mimetypes = ['application/x-tf', 'application/x-terraform']
    tokens = {'root': [include('string'), include('punctuation'), include('curly'), include('basic'), include('whitespace'), ('[0-9]+', Number)], 'basic': [(words(('true', 'false'), prefix='\\b', suffix='\\b'), Keyword.Type), ('\\s*/\\*', Comment.Multiline, 'comment'), ('\\s*#.*\\n', Comment.Single), ('(.*?)(\\s*)(=)', bygroups(Name.Attribute, Text, Operator)), (words(('variable', 'resource', 'provider', 'provisioner', 'module'), prefix='\\b', suffix='\\b'), Keyword.Reserved, 'function'), (words(('ingress', 'egress', 'listener', 'default', 'connection'), prefix='\\b', suffix='\\b'), Keyword.Declaration), ('\\$\\{', String.Interpol, 'var_builtin')], 'function': [('(\\s+)(".*")(\\s+)', bygroups(Text, String, Text)), include('punctuation'), include('curly')], 'var_builtin': [('\\$\\{', String.Interpol, '#push'), (words(('concat', 'file', 'join', 'lookup', 'element'), prefix='\\b', suffix='\\b'), Name.Builtin), include('string'), include('punctuation'), ('\\s+', Text), ('\\}', String.Interpol, '#pop')], 'string': [('(".*")', bygroups(String.Double))], 'punctuation': [('[\\[\\](),.]', Punctuation)], 'curly': [('\\{', Text.Punctuation), ('\\}', Text.Punctuation)], 'comment': [('[^*/]', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)], 'whitespace': [('\\n', Text), ('\\s+', Text), ('\\\\\\n', Text)]}