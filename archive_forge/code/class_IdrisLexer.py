import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class IdrisLexer(RegexLexer):
    """
    A lexer for the dependently typed programming language Idris.

    Based on the Haskell and Agda Lexer.

    .. versionadded:: 2.0
    """
    name = 'Idris'
    aliases = ['idris', 'idr']
    filenames = ['*.idr']
    mimetypes = ['text/x-idris']
    reserved = ('case', 'class', 'data', 'default', 'using', 'do', 'else', 'if', 'in', 'infix[lr]?', 'instance', 'rewrite', 'auto', 'namespace', 'codata', 'mutual', 'private', 'public', 'abstract', 'total', 'partial', 'let', 'proof', 'of', 'then', 'static', 'where', '_', 'with', 'pattern', 'term', 'syntax', 'prefix', 'postulate', 'parameters', 'record', 'dsl', 'impossible', 'implicit', 'tactics', 'intros', 'intro', 'compute', 'refine', 'exact', 'trivial')
    ascii = ('NUL', 'SOH', '[SE]TX', 'EOT', 'ENQ', 'ACK', 'BEL', 'BS', 'HT', 'LF', 'VT', 'FF', 'CR', 'S[OI]', 'DLE', 'DC[1-4]', 'NAK', 'SYN', 'ETB', 'CAN', 'EM', 'SUB', 'ESC', '[FGRU]S', 'SP', 'DEL')
    directives = ('lib', 'link', 'flag', 'include', 'hide', 'freeze', 'access', 'default', 'logging', 'dynamic', 'name', 'error_handlers', 'language')
    tokens = {'root': [('^(\\s*)(%%%s)' % '|'.join(directives), bygroups(Text, Keyword.Reserved)), ('(\\s*)(--(?![!#$%&*+./<=>?@^|_~:\\\\]).*?)$', bygroups(Text, Comment.Single)), ('(\\s*)(\\|{3}.*?)$', bygroups(Text, Comment.Single)), ('(\\s*)(\\{-)', bygroups(Text, Comment.Multiline), 'comment'), ('^(\\s*)([^\\s(){}]+)(\\s*)(:)(\\s*)', bygroups(Text, Name.Function, Text, Operator.Word, Text)), ("\\b(%s)(?!\\')\\b" % '|'.join(reserved), Keyword.Reserved), ('(import|module)(\\s+)', bygroups(Keyword.Reserved, Text), 'module'), ("('')?[A-Z][\\w\\']*", Keyword.Type), ("[a-z][\\w\\']*", Text), ('(<-|::|->|=>|=)', Operator.Word), ('([(){}\\[\\]:!#$%&*+.\\\\/<=>?@^|~-]+)', Operator.Word), ('\\d+[eE][+-]?\\d+', Number.Float), ('\\d+\\.\\d+([eE][+-]?\\d+)?', Number.Float), ('0[xX][\\da-fA-F]+', Number.Hex), ('\\d+', Number.Integer), ("'", String.Char, 'character'), ('"', String, 'string'), ('[^\\s(){}]+', Text), ('\\s+?', Text)], 'module': [('\\s+', Text), ('([A-Z][\\w.]*)(\\s+)(\\()', bygroups(Name.Namespace, Text, Punctuation), 'funclist'), ('[A-Z][\\w.]*', Name.Namespace, '#pop')], 'funclist': [('\\s+', Text), ('[A-Z]\\w*', Keyword.Type), ("(_[\\w\\']+|[a-z][\\w\\']*)", Name.Function), ('--.*$', Comment.Single), ('\\{-', Comment.Multiline, 'comment'), (',', Punctuation), ('[:!#$%&*+.\\\\/<=>?@^|~-]+', Operator), ('\\(', Punctuation, ('funclist', 'funclist')), ('\\)', Punctuation, '#pop:2')], 'comment': [('[^-{}]+', Comment.Multiline), ('\\{-', Comment.Multiline, '#push'), ('-\\}', Comment.Multiline, '#pop'), ('[-{}]', Comment.Multiline)], 'character': [("[^\\\\']", String.Char), ('\\\\', String.Escape, 'escape'), ("'", String.Char, '#pop')], 'string': [('[^\\\\"]+', String), ('\\\\', String.Escape, 'escape'), ('"', String, '#pop')], 'escape': [('[abfnrtv"\\\'&\\\\]', String.Escape, '#pop'), ('\\^[][A-Z@^_]', String.Escape, '#pop'), ('|'.join(ascii), String.Escape, '#pop'), ('o[0-7]+', String.Escape, '#pop'), ('x[\\da-fA-F]+', String.Escape, '#pop'), ('\\d+', String.Escape, '#pop'), ('\\s+\\\\', String.Escape, '#pop')]}