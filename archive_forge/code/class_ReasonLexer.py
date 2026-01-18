import re
from pygments.lexer import RegexLexer, include, bygroups, default, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class ReasonLexer(RegexLexer):
    """
    For the ReasonML language.

    .. versionadded:: 2.6
    """
    name = 'ReasonML'
    url = 'https://reasonml.github.io/'
    aliases = ['reasonml', 'reason']
    filenames = ['*.re', '*.rei']
    mimetypes = ['text/x-reasonml']
    keywords = ('as', 'assert', 'begin', 'class', 'constraint', 'do', 'done', 'downto', 'else', 'end', 'exception', 'external', 'false', 'for', 'fun', 'esfun', 'function', 'functor', 'if', 'in', 'include', 'inherit', 'initializer', 'lazy', 'let', 'switch', 'module', 'pub', 'mutable', 'new', 'nonrec', 'object', 'of', 'open', 'pri', 'rec', 'sig', 'struct', 'then', 'to', 'true', 'try', 'type', 'val', 'virtual', 'when', 'while', 'with')
    keyopts = ('!=', '#', '&', '&&', '\\(', '\\)', '\\*', '\\+', ',', '-', '-\\.', '=>', '\\.', '\\.\\.', '\\.\\.\\.', ':', '::', ':=', ':>', ';', ';;', '<', '<-', '=', '>', '>]', '>\\}', '\\?', '\\?\\?', '\\[', '\\[<', '\\[>', '\\[\\|', ']', '_', '`', '\\{', '\\{<', '\\|', '\\|\\|', '\\|]', '\\}', '~')
    operators = '[!$%&*+\\./:<=>?@^|~-]'
    word_operators = ('and', 'asr', 'land', 'lor', 'lsl', 'lsr', 'lxor', 'mod', 'or')
    prefix_syms = '[!?~]'
    infix_syms = '[=<>@^|&+\\*/$%-]'
    primitives = ('unit', 'int', 'float', 'bool', 'string', 'char', 'list', 'array')
    tokens = {'escape-sequence': [('\\\\[\\\\"\\\'ntbr]', String.Escape), ('\\\\[0-9]{3}', String.Escape), ('\\\\x[0-9a-fA-F]{2}', String.Escape)], 'root': [('\\s+', Text), ('false|true|\\(\\)|\\[\\]', Name.Builtin.Pseudo), ("\\b([A-Z][\\w\\']*)(?=\\s*\\.)", Name.Namespace, 'dotted'), ("\\b([A-Z][\\w\\']*)", Name.Class), ('//.*?\\n', Comment.Single), ('\\/\\*(?!/)', Comment.Multiline, 'comment'), ('\\b(%s)\\b' % '|'.join(keywords), Keyword), ('(%s)' % '|'.join(keyopts[::-1]), Operator.Word), ('(%s|%s)?%s' % (infix_syms, prefix_syms, operators), Operator), ('\\b(%s)\\b' % '|'.join(word_operators), Operator.Word), ('\\b(%s)\\b' % '|'.join(primitives), Keyword.Type), ("[^\\W\\d][\\w']*", Name), ('-?\\d[\\d_]*(.[\\d_]*)?([eE][+\\-]?\\d[\\d_]*)', Number.Float), ('0[xX][\\da-fA-F][\\da-fA-F_]*', Number.Hex), ('0[oO][0-7][0-7_]*', Number.Oct), ('0[bB][01][01_]*', Number.Bin), ('\\d[\\d_]*', Number.Integer), ('\'(?:(\\\\[\\\\\\"\'ntbr ])|(\\\\[0-9]{3})|(\\\\x[0-9a-fA-F]{2}))\'', String.Char), ("'.'", String.Char), ("'", Keyword), ('"', String.Double, 'string'), ("[~?][a-z][\\w\\']*:", Name.Variable)], 'comment': [('[^/*]+', Comment.Multiline), ('\\/\\*', Comment.Multiline, '#push'), ('\\*\\/', Comment.Multiline, '#pop'), ('\\*', Comment.Multiline)], 'string': [('[^\\\\"]+', String.Double), include('escape-sequence'), ('\\\\\\n', String.Double), ('"', String.Double, '#pop')], 'dotted': [('\\s+', Text), ('\\.', Punctuation), ("[A-Z][\\w\\']*(?=\\s*\\.)", Name.Namespace), ("[A-Z][\\w\\']*", Name.Class, '#pop'), ("[a-z_][\\w\\']*", Name, '#pop'), default('#pop')]}