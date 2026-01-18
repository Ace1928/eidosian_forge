from pygments.lexer import RegexLexer, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class SnobolLexer(RegexLexer):
    """
    Lexer for the SNOBOL4 programming language.

    Recognizes the common ASCII equivalents of the original SNOBOL4 operators.
    Does not require spaces around binary operators.

    .. versionadded:: 1.5
    """
    name = 'Snobol'
    aliases = ['snobol']
    filenames = ['*.snobol']
    mimetypes = ['text/x-snobol']
    tokens = {'root': [('\\*.*\\n', Comment), ('[+.] ', Punctuation, 'statement'), ('-.*\\n', Comment), ('END\\s*\\n', Name.Label, 'heredoc'), ('[A-Za-z$][\\w$]*', Name.Label, 'statement'), ('\\s+', Text, 'statement')], 'statement': [('\\s*\\n', Text, '#pop'), ('\\s+', Text), ('(?<=[^\\w.])(LT|LE|EQ|NE|GE|GT|INTEGER|IDENT|DIFFER|LGT|SIZE|REPLACE|TRIM|DUPL|REMDR|DATE|TIME|EVAL|APPLY|OPSYN|LOAD|UNLOAD|LEN|SPAN|BREAK|ANY|NOTANY|TAB|RTAB|REM|POS|RPOS|FAIL|FENCE|ABORT|ARB|ARBNO|BAL|SUCCEED|INPUT|OUTPUT|TERMINAL)(?=[^\\w.])', Name.Builtin), ('[A-Za-z][\\w.]*', Name), ('\\*\\*|[?$.!%*/#+\\-@|&\\\\=]', Operator), ('"[^"]*"', String), ("'[^']*'", String), ('[0-9]+(?=[^.EeDd])', Number.Integer), ('[0-9]+(\\.[0-9]*)?([EDed][-+]?[0-9]+)?', Number.Float), (':', Punctuation, 'goto'), ('[()<>,;]', Punctuation)], 'goto': [('\\s*\\n', Text, '#pop:2'), ('\\s+', Text), ('F|S', Keyword), ('(\\()([A-Za-z][\\w.]*)(\\))', bygroups(Punctuation, Name.Label, Punctuation))], 'heredoc': [('.*\\n', String.Heredoc)]}