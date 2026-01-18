import re
from pygments.lexer import RegexLexer, DelegatingLexer, bygroups, include, \
from pygments.token import Punctuation, \
from pygments.util import get_choice_opt, iteritems
from pygments import unistring as uni
from pygments.lexers.html import XmlLexer
class FSharpLexer(RegexLexer):
    """
    For the F# language (version 3.0).

    AAAAACK Strings
    http://research.microsoft.com/en-us/um/cambridge/projects/fsharp/manual/spec.html#_Toc335818775

    .. versionadded:: 1.5
    """
    name = 'FSharp'
    aliases = ['fsharp']
    filenames = ['*.fs', '*.fsi']
    mimetypes = ['text/x-fsharp']
    keywords = ['abstract', 'as', 'assert', 'base', 'begin', 'class', 'default', 'delegate', 'do!', 'do', 'done', 'downcast', 'downto', 'elif', 'else', 'end', 'exception', 'extern', 'false', 'finally', 'for', 'function', 'fun', 'global', 'if', 'inherit', 'inline', 'interface', 'internal', 'in', 'lazy', 'let!', 'let', 'match', 'member', 'module', 'mutable', 'namespace', 'new', 'null', 'of', 'open', 'override', 'private', 'public', 'rec', 'return!', 'return', 'select', 'static', 'struct', 'then', 'to', 'true', 'try', 'type', 'upcast', 'use!', 'use', 'val', 'void', 'when', 'while', 'with', 'yield!', 'yield']
    keywords += ['atomic', 'break', 'checked', 'component', 'const', 'constraint', 'constructor', 'continue', 'eager', 'event', 'external', 'fixed', 'functor', 'include', 'method', 'mixin', 'object', 'parallel', 'process', 'protected', 'pure', 'sealed', 'tailcall', 'trait', 'virtual', 'volatile']
    keyopts = ['!=', '#', '&&', '&', '\\(', '\\)', '\\*', '\\+', ',', '-\\.', '->', '-', '\\.\\.', '\\.', '::', ':=', ':>', ':', ';;', ';', '<-', '<\\]', '<', '>\\]', '>', '\\?\\?', '\\?', '\\[<', '\\[\\|', '\\[', '\\]', '_', '`', '\\{', '\\|\\]', '\\|', '\\}', '~', '<@@', '<@', '=', '@>', '@@>']
    operators = '[!$%&*+\\./:<=>?@^|~-]'
    word_operators = ['and', 'or', 'not']
    prefix_syms = '[!?~]'
    infix_syms = '[=<>@^|&+\\*/$%-]'
    primitives = ['sbyte', 'byte', 'char', 'nativeint', 'unativeint', 'float32', 'single', 'float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'decimal', 'unit', 'bool', 'string', 'list', 'exn', 'obj', 'enum']
    tokens = {'escape-sequence': [('\\\\[\\\\"\\\'ntbrafv]', String.Escape), ('\\\\[0-9]{3}', String.Escape), ('\\\\u[0-9a-fA-F]{4}', String.Escape), ('\\\\U[0-9a-fA-F]{8}', String.Escape)], 'root': [('\\s+', Text), ('\\(\\)|\\[\\]', Name.Builtin.Pseudo), ("\\b(?<!\\.)([A-Z][\\w\\']*)(?=\\s*\\.)", Name.Namespace, 'dotted'), ("\\b([A-Z][\\w\\']*)", Name), ('///.*?\\n', String.Doc), ('//.*?\\n', Comment.Single), ('\\(\\*(?!\\))', Comment, 'comment'), ('@"', String, 'lstring'), ('"""', String, 'tqs'), ('"', String, 'string'), ('\\b(open|module)(\\s+)([\\w.]+)', bygroups(Keyword, Text, Name.Namespace)), ('\\b(let!?)(\\s+)(\\w+)', bygroups(Keyword, Text, Name.Variable)), ('\\b(type)(\\s+)(\\w+)', bygroups(Keyword, Text, Name.Class)), ('\\b(member|override)(\\s+)(\\w+)(\\.)(\\w+)', bygroups(Keyword, Text, Name, Punctuation, Name.Function)), ('\\b(%s)\\b' % '|'.join(keywords), Keyword), ('``([^`\\n\\r\\t]|`[^`\\n\\r\\t])+``', Name), ('(%s)' % '|'.join(keyopts), Operator), ('(%s|%s)?%s' % (infix_syms, prefix_syms, operators), Operator), ('\\b(%s)\\b' % '|'.join(word_operators), Operator.Word), ('\\b(%s)\\b' % '|'.join(primitives), Keyword.Type), ('#[ \\t]*(if|endif|else|line|nowarn|light|\\d+)\\b.*?\\n', Comment.Preproc), ("[^\\W\\d][\\w']*", Name), ('\\d[\\d_]*[uU]?[yslLnQRZINGmM]?', Number.Integer), ('0[xX][\\da-fA-F][\\da-fA-F_]*[uU]?[yslLn]?[fF]?', Number.Hex), ('0[oO][0-7][0-7_]*[uU]?[yslLn]?', Number.Oct), ('0[bB][01][01_]*[uU]?[yslLn]?', Number.Bin), ('-?\\d[\\d_]*(.[\\d_]*)?([eE][+\\-]?\\d[\\d_]*)[fFmM]?', Number.Float), ('\'(?:(\\\\[\\\\\\"\'ntbr ])|(\\\\[0-9]{3})|(\\\\x[0-9a-fA-F]{2}))\'B?', String.Char), ("'.'", String.Char), ("'", Keyword), ('@?"', String.Double, 'string'), ("[~?][a-z][\\w\\']*:", Name.Variable)], 'dotted': [('\\s+', Text), ('\\.', Punctuation), ("[A-Z][\\w\\']*(?=\\s*\\.)", Name.Namespace), ("[A-Z][\\w\\']*", Name, '#pop'), ("[a-z_][\\w\\']*", Name, '#pop'), default('#pop')], 'comment': [('[^(*)@"]+', Comment), ('\\(\\*', Comment, '#push'), ('\\*\\)', Comment, '#pop'), ('@"', String, 'lstring'), ('"""', String, 'tqs'), ('"', String, 'string'), ('[(*)@]', Comment)], 'string': [('[^\\\\"]+', String), include('escape-sequence'), ('\\\\\\n', String), ('\\n', String), ('"B?', String, '#pop')], 'lstring': [('[^"]+', String), ('\\n', String), ('""', String), ('"B?', String, '#pop')], 'tqs': [('[^"]+', String), ('\\n', String), ('"""B?', String, '#pop'), ('"', String)]}