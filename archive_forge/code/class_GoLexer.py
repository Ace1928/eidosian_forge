import re
from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class GoLexer(RegexLexer):
    """
    For `Go <http://golang.org>`_ source.

    .. versionadded:: 1.2
    """
    name = 'Go'
    filenames = ['*.go']
    aliases = ['go']
    mimetypes = ['text/x-gosrc']
    flags = re.MULTILINE | re.UNICODE
    tokens = {'root': [('\\n', Text), ('\\s+', Text), ('\\\\\\n', Text), ('//(.*?)\\n', Comment.Single), ('/(\\\\\\n)?[*](.|\\n)*?[*](\\\\\\n)?/', Comment.Multiline), ('(import|package)\\b', Keyword.Namespace), ('(var|func|struct|map|chan|type|interface|const)\\b', Keyword.Declaration), (words(('break', 'default', 'select', 'case', 'defer', 'go', 'else', 'goto', 'switch', 'fallthrough', 'if', 'range', 'continue', 'for', 'return'), suffix='\\b'), Keyword), ('(true|false|iota|nil)\\b', Keyword.Constant), (words(('uint', 'uint8', 'uint16', 'uint32', 'uint64', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64', 'complex64', 'complex128', 'byte', 'rune', 'string', 'bool', 'error', 'uintptr', 'print', 'println', 'panic', 'recover', 'close', 'complex', 'real', 'imag', 'len', 'cap', 'append', 'copy', 'delete', 'new', 'make'), suffix='\\b(\\()'), bygroups(Name.Builtin, Punctuation)), (words(('uint', 'uint8', 'uint16', 'uint32', 'uint64', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64', 'complex64', 'complex128', 'byte', 'rune', 'string', 'bool', 'error', 'uintptr'), suffix='\\b'), Keyword.Type), ('\\d+i', Number), ('\\d+\\.\\d*([Ee][-+]\\d+)?i', Number), ('\\.\\d+([Ee][-+]\\d+)?i', Number), ('\\d+[Ee][-+]\\d+i', Number), ('\\d+(\\.\\d+[eE][+\\-]?\\d+|\\.\\d*|[eE][+\\-]?\\d+)', Number.Float), ('\\.\\d+([eE][+\\-]?\\d+)?', Number.Float), ('0[0-7]+', Number.Oct), ('0[xX][0-9a-fA-F]+', Number.Hex), ('(0|[1-9][0-9]*)', Number.Integer), ('\'(\\\\[\'"\\\\abfnrtv]|\\\\x[0-9a-fA-F]{2}|\\\\[0-7]{1,3}|\\\\u[0-9a-fA-F]{4}|\\\\U[0-9a-fA-F]{8}|[^\\\\])\'', String.Char), ('`[^`]*`', String), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ('(<<=|>>=|<<|>>|<=|>=|&\\^=|&\\^|\\+=|-=|\\*=|/=|%=|&=|\\|=|&&|\\|\\||<-|\\+\\+|--|==|!=|:=|\\.\\.\\.|[+\\-*/%&])', Operator), ('[|^<>=!()\\[\\]{}.,;:]', Punctuation), ('[^\\W\\d]\\w*', Name.Other)]}