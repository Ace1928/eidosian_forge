import re
from pygments.lexer import RegexLexer, include, bygroups, using, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.html import HtmlLexer
from pygments.lexers import _stan_builtins
class StanLexer(RegexLexer):
    """Pygments Lexer for Stan models.

    The Stan modeling language is specified in the *Stan Modeling Language
    User's Guide and Reference Manual, v2.8.0*,
    `pdf <https://github.com/stan-dev/stan/releases/download/v2.8.8/stan-reference-2.8.0.pdf>`__.

    .. versionadded:: 1.6
    """
    name = 'Stan'
    aliases = ['stan']
    filenames = ['*.stan']
    tokens = {'whitespace': [('\\s+', Text)], 'comments': [('(?s)/\\*.*?\\*/', Comment.Multiline), ('(//|#).*$', Comment.Single)], 'root': [('"[^"]*"', String), include('comments'), include('whitespace'), ('(%s)(\\s*)(\\{)' % '|'.join(('functions', 'data', 'transformed\\s+?data', 'parameters', 'transformed\\s+parameters', 'model', 'generated\\s+quantities')), bygroups(Keyword.Namespace, Text, Punctuation)), ('(%s)\\b' % '|'.join(_stan_builtins.KEYWORDS), Keyword), ('T(?=\\s*\\[)', Keyword), ('(%s)\\b' % '|'.join(_stan_builtins.TYPES), Keyword.Type), ('[;:,\\[\\]()]', Punctuation), ('(%s)(?=\\s*\\()' % '|'.join(_stan_builtins.FUNCTIONS + _stan_builtins.DISTRIBUTIONS), Name.Builtin), ('[A-Za-z]\\w*__\\b', Name.Builtin.Pseudo), ('(%s)\\b' % '|'.join(_stan_builtins.RESERVED), Keyword.Reserved), ('[A-Za-z]\\w*(?=\\s*\\()]', Name.Function), ('[A-Za-z]\\w*\\b', Name), ('-?[0-9]+(\\.[0-9]+)?[eE]-?[0-9]+', Number.Float), ('-?[0-9]*\\.[0-9]*', Number.Float), ('-?[0-9]+', Number.Integer), ('<-|~', Operator), ("\\+|-|\\.?\\*|\\.?/|\\\\|'|\\^|==?|!=?|<=?|>=?|\\|\\||&&", Operator), ('[{}]', Punctuation)]}

    def analyse_text(text):
        if re.search('^\\s*parameters\\s*\\{', text, re.M):
            return 1.0
        else:
            return 0.0