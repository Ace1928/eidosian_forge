import re
from pygments.lexer import RegexLexer, include, bygroups, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
class SchemeLexer(RegexLexer):
    """
    A Scheme lexer, parsing a stream and outputting the tokens
    needed to highlight scheme code.
    This lexer could be most probably easily subclassed to parse
    other LISP-Dialects like Common Lisp, Emacs Lisp or AutoLisp.

    This parser is checked with pastes from the LISP pastebin
    at http://paste.lisp.org/ to cover as much syntax as possible.

    It supports the full Scheme syntax as defined in R5RS.

    .. versionadded:: 0.6
    """
    name = 'Scheme'
    aliases = ['scheme', 'scm']
    filenames = ['*.scm', '*.ss']
    mimetypes = ['text/x-scheme', 'application/x-scheme']
    keywords = ('lambda', 'define', 'if', 'else', 'cond', 'and', 'or', 'case', 'let', 'let*', 'letrec', 'begin', 'do', 'delay', 'set!', '=>', 'quote', 'quasiquote', 'unquote', 'unquote-splicing', 'define-syntax', 'let-syntax', 'letrec-syntax', 'syntax-rules')
    builtins = ('*', '+', '-', '/', '<', '<=', '=', '>', '>=', 'abs', 'acos', 'angle', 'append', 'apply', 'asin', 'assoc', 'assq', 'assv', 'atan', 'boolean?', 'caaaar', 'caaadr', 'caaar', 'caadar', 'caaddr', 'caadr', 'caar', 'cadaar', 'cadadr', 'cadar', 'caddar', 'cadddr', 'caddr', 'cadr', 'call-with-current-continuation', 'call-with-input-file', 'call-with-output-file', 'call-with-values', 'call/cc', 'car', 'cdaaar', 'cdaadr', 'cdaar', 'cdadar', 'cdaddr', 'cdadr', 'cdar', 'cddaar', 'cddadr', 'cddar', 'cdddar', 'cddddr', 'cdddr', 'cddr', 'cdr', 'ceiling', 'char->integer', 'char-alphabetic?', 'char-ci<=?', 'char-ci<?', 'char-ci=?', 'char-ci>=?', 'char-ci>?', 'char-downcase', 'char-lower-case?', 'char-numeric?', 'char-ready?', 'char-upcase', 'char-upper-case?', 'char-whitespace?', 'char<=?', 'char<?', 'char=?', 'char>=?', 'char>?', 'char?', 'close-input-port', 'close-output-port', 'complex?', 'cons', 'cos', 'current-input-port', 'current-output-port', 'denominator', 'display', 'dynamic-wind', 'eof-object?', 'eq?', 'equal?', 'eqv?', 'eval', 'even?', 'exact->inexact', 'exact?', 'exp', 'expt', 'floor', 'for-each', 'force', 'gcd', 'imag-part', 'inexact->exact', 'inexact?', 'input-port?', 'integer->char', 'integer?', 'interaction-environment', 'lcm', 'length', 'list', 'list->string', 'list->vector', 'list-ref', 'list-tail', 'list?', 'load', 'log', 'magnitude', 'make-polar', 'make-rectangular', 'make-string', 'make-vector', 'map', 'max', 'member', 'memq', 'memv', 'min', 'modulo', 'negative?', 'newline', 'not', 'null-environment', 'null?', 'number->string', 'number?', 'numerator', 'odd?', 'open-input-file', 'open-output-file', 'output-port?', 'pair?', 'peek-char', 'port?', 'positive?', 'procedure?', 'quotient', 'rational?', 'rationalize', 'read', 'read-char', 'real-part', 'real?', 'remainder', 'reverse', 'round', 'scheme-report-environment', 'set-car!', 'set-cdr!', 'sin', 'sqrt', 'string', 'string->list', 'string->number', 'string->symbol', 'string-append', 'string-ci<=?', 'string-ci<?', 'string-ci=?', 'string-ci>=?', 'string-ci>?', 'string-copy', 'string-fill!', 'string-length', 'string-ref', 'string-set!', 'string<=?', 'string<?', 'string=?', 'string>=?', 'string>?', 'string?', 'substring', 'symbol->string', 'symbol?', 'tan', 'transcript-off', 'transcript-on', 'truncate', 'values', 'vector', 'vector->list', 'vector-fill!', 'vector-length', 'vector-ref', 'vector-set!', 'vector?', 'with-input-from-file', 'with-output-to-file', 'write', 'write-char', 'zero?')
    valid_name = '[\\w!$%&*+,/:<=>?@^~|-]+'
    tokens = {'root': [(';.*$', Comment.Single), ('#\\|', Comment.Multiline, 'multiline-comment'), ('#;\\s*\\(', Comment, 'commented-form'), ('#!r6rs', Comment), ('\\s+', Text), ('-?\\d+\\.\\d+', Number.Float), ('-?\\d+', Number.Integer), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ("'" + valid_name, String.Symbol), ('#\\\\([()/\'\\"._!§$%& ?=+-]|[a-zA-Z0-9]+)', String.Char), ('(#t|#f)', Name.Constant), ("('|#|`|,@|,|\\.)", Operator), ('(%s)' % '|'.join((re.escape(entry) + ' ' for entry in keywords)), Keyword), ("(?<='\\()" + valid_name, Name.Variable), ('(?<=#\\()' + valid_name, Name.Variable), ('(?<=\\()(%s)' % '|'.join((re.escape(entry) + ' ' for entry in builtins)), Name.Builtin), ('(?<=\\()' + valid_name, Name.Function), (valid_name, Name.Variable), ('(\\(|\\))', Punctuation), ('(\\[|\\])', Punctuation)], 'multiline-comment': [('#\\|', Comment.Multiline, '#push'), ('\\|#', Comment.Multiline, '#pop'), ('[^|#]+', Comment.Multiline), ('[|#]', Comment.Multiline)], 'commented-form': [('\\(', Comment, '#push'), ('\\)', Comment, '#pop'), ('[^()]+', Comment)]}