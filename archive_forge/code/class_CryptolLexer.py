import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class CryptolLexer(RegexLexer):
    """
    FIXME: A Cryptol2 lexer based on the lexemes defined in the Haskell 98 Report.

    .. versionadded:: 2.0
    """
    name = 'Cryptol'
    aliases = ['cryptol', 'cry']
    filenames = ['*.cry']
    mimetypes = ['text/x-cryptol']
    reserved = ('Arith', 'Bit', 'Cmp', 'False', 'Inf', 'True', 'else', 'export', 'extern', 'fin', 'if', 'import', 'inf', 'lg2', 'max', 'min', 'module', 'newtype', 'pragma', 'property', 'then', 'type', 'where', 'width')
    ascii = ('NUL', 'SOH', '[SE]TX', 'EOT', 'ENQ', 'ACK', 'BEL', 'BS', 'HT', 'LF', 'VT', 'FF', 'CR', 'S[OI]', 'DLE', 'DC[1-4]', 'NAK', 'SYN', 'ETB', 'CAN', 'EM', 'SUB', 'ESC', '[FGRU]S', 'SP', 'DEL')
    tokens = {'root': [('\\s+', Text), ('//.*$', Comment.Single), ('/\\*', Comment.Multiline, 'comment'), ('\\bimport\\b', Keyword.Reserved, 'import'), ('\\bmodule\\b', Keyword.Reserved, 'module'), ('\\berror\\b', Name.Exception), ("\\b(%s)(?!\\')\\b" % '|'.join(reserved), Keyword.Reserved), ("^[_a-z][\\w\\']*", Name.Function), ("'?[_a-z][\\w']*", Name), ("('')?[A-Z][\\w\\']*", Keyword.Type), ('\\\\(?![:!#$%&*+.\\\\/<=>?@^|~-]+)', Name.Function), ('(<-|::|->|=>|=)(?![:!#$%&*+.\\\\/<=>?@^|~-]+)', Operator.Word), (':[:!#$%&*+.\\\\/<=>?@^|~-]*', Keyword.Type), ('[:!#$%&*+.\\\\/<=>?@^|~-]+', Operator), ('\\d+[eE][+-]?\\d+', Number.Float), ('\\d+\\.\\d+([eE][+-]?\\d+)?', Number.Float), ('0[oO][0-7]+', Number.Oct), ('0[xX][\\da-fA-F]+', Number.Hex), ('\\d+', Number.Integer), ("'", String.Char, 'character'), ('"', String, 'string'), ('\\[\\]', Keyword.Type), ('\\(\\)', Name.Builtin), ('[][(),;`{}]', Punctuation)], 'import': [('\\s+', Text), ('"', String, 'string'), ('\\)', Punctuation, '#pop'), ('qualified\\b', Keyword), ('([A-Z][\\w.]*)(\\s+)(as)(\\s+)([A-Z][\\w.]*)', bygroups(Name.Namespace, Text, Keyword, Text, Name), '#pop'), ('([A-Z][\\w.]*)(\\s+)(hiding)(\\s+)(\\()', bygroups(Name.Namespace, Text, Keyword, Text, Punctuation), 'funclist'), ('([A-Z][\\w.]*)(\\s+)(\\()', bygroups(Name.Namespace, Text, Punctuation), 'funclist'), ('[\\w.]+', Name.Namespace, '#pop')], 'module': [('\\s+', Text), ('([A-Z][\\w.]*)(\\s+)(\\()', bygroups(Name.Namespace, Text, Punctuation), 'funclist'), ('[A-Z][\\w.]*', Name.Namespace, '#pop')], 'funclist': [('\\s+', Text), ('[A-Z]\\w*', Keyword.Type), ("(_[\\w\\']+|[a-z][\\w\\']*)", Name.Function), (',', Punctuation), ('[:!#$%&*+.\\\\/<=>?@^|~-]+', Operator), ('\\(', Punctuation, ('funclist', 'funclist')), ('\\)', Punctuation, '#pop:2')], 'comment': [('[^/*]+', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)], 'character': [("[^\\\\']'", String.Char, '#pop'), ('\\\\', String.Escape, 'escape'), ("'", String.Char, '#pop')], 'string': [('[^\\\\"]+', String), ('\\\\', String.Escape, 'escape'), ('"', String, '#pop')], 'escape': [('[abfnrtv"\\\'&\\\\]', String.Escape, '#pop'), ('\\^[][A-Z@^_]', String.Escape, '#pop'), ('|'.join(ascii), String.Escape, '#pop'), ('o[0-7]+', String.Escape, '#pop'), ('x[\\da-fA-F]+', String.Escape, '#pop'), ('\\d+', String.Escape, '#pop'), ('\\s+\\\\', String.Escape, '#pop')]}
    EXTRA_KEYWORDS = set(('join', 'split', 'reverse', 'transpose', 'width', 'length', 'tail', '<<', '>>', '<<<', '>>>', 'const', 'reg', 'par', 'seq', 'ASSERT', 'undefined', 'error', 'trace'))

    def get_tokens_unprocessed(self, text):
        stack = ['root']
        for index, token, value in RegexLexer.get_tokens_unprocessed(self, text, stack):
            if token is Name and value in self.EXTRA_KEYWORDS:
                yield (index, Name.Builtin, value)
            else:
                yield (index, token, value)