import re
from pygments.lexer import RegexLexer, include, bygroups, default, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class FStarLexer(RegexLexer):
    """
    For the F* language.
    .. versionadded:: 2.7
    """
    name = 'FStar'
    url = 'https://www.fstar-lang.org/'
    aliases = ['fstar']
    filenames = ['*.fst', '*.fsti']
    mimetypes = ['text/x-fstar']
    keywords = ('abstract', 'attributes', 'noeq', 'unopteq', 'andbegin', 'by', 'default', 'effect', 'else', 'end', 'ensures', 'exception', 'exists', 'false', 'forall', 'fun', 'function', 'if', 'in', 'include', 'inline', 'inline_for_extraction', 'irreducible', 'logic', 'match', 'module', 'mutable', 'new', 'new_effect', 'noextract', 'of', 'open', 'opaque', 'private', 'range_of', 'reifiable', 'reify', 'reflectable', 'requires', 'set_range_of', 'sub_effect', 'synth', 'then', 'total', 'true', 'try', 'type', 'unfold', 'unfoldable', 'val', 'when', 'with', 'not')
    decl_keywords = ('let', 'rec')
    assume_keywords = ('assume', 'admit', 'assert', 'calc')
    keyopts = ('~', '-', '/\\\\', '\\\\/', '<:', '<@', '\\(\\|', '\\|\\)', '#', 'u#', '&', '\\(', '\\)', '\\(\\)', ',', '~>', '->', '<-', '<--', '<==>', '==>', '\\.', '\\?', '\\?\\.', '\\.\\[', '\\.\\(', '\\.\\(\\|', '\\.\\[\\|', '\\{:pattern', ':', '::', ':=', ';', ';;', '=', '%\\[', '!\\{', '\\[', '\\[@', '\\[\\|', '\\|>', '\\]', '\\|\\]', '\\{', '\\|', '\\}', '\\$')
    operators = '[!$%&*+\\./:<=>?@^|~-]'
    prefix_syms = '[!?~]'
    infix_syms = '[=<>@^|&+\\*/$%-]'
    primitives = ('unit', 'int', 'float', 'bool', 'string', 'char', 'list', 'array')
    tokens = {'escape-sequence': [('\\\\[\\\\"\\\'ntbr]', String.Escape), ('\\\\[0-9]{3}', String.Escape), ('\\\\x[0-9a-fA-F]{2}', String.Escape)], 'root': [('\\s+', Text), ('false|true|False|True|\\(\\)|\\[\\]', Name.Builtin.Pseudo), ("\\b([A-Z][\\w\\']*)(?=\\s*\\.)", Name.Namespace, 'dotted'), ("\\b([A-Z][\\w\\']*)", Name.Class), ('\\(\\*(?![)])', Comment, 'comment'), ('\\/\\/.+$', Comment), ('\\b(%s)\\b' % '|'.join(keywords), Keyword), ('\\b(%s)\\b' % '|'.join(assume_keywords), Name.Exception), ('\\b(%s)\\b' % '|'.join(decl_keywords), Keyword.Declaration), ('(%s)' % '|'.join(keyopts[::-1]), Operator), ('(%s|%s)?%s' % (infix_syms, prefix_syms, operators), Operator), ('\\b(%s)\\b' % '|'.join(primitives), Keyword.Type), ("[^\\W\\d][\\w']*", Name), ('-?\\d[\\d_]*(.[\\d_]*)?([eE][+\\-]?\\d[\\d_]*)', Number.Float), ('0[xX][\\da-fA-F][\\da-fA-F_]*', Number.Hex), ('0[oO][0-7][0-7_]*', Number.Oct), ('0[bB][01][01_]*', Number.Bin), ('\\d[\\d_]*', Number.Integer), ('\'(?:(\\\\[\\\\\\"\'ntbr ])|(\\\\[0-9]{3})|(\\\\x[0-9a-fA-F]{2}))\'', String.Char), ("'.'", String.Char), ("'", Keyword), ("\\`([\\w\\'.]+)\\`", Operator.Word), ('\\`', Keyword), ('"', String.Double, 'string'), ("[~?][a-z][\\w\\']*:", Name.Variable)], 'comment': [('[^(*)]+', Comment), ('\\(\\*', Comment, '#push'), ('\\*\\)', Comment, '#pop'), ('[(*)]', Comment)], 'string': [('[^\\\\"]+', String.Double), include('escape-sequence'), ('\\\\\\n', String.Double), ('"', String.Double, '#pop')], 'dotted': [('\\s+', Text), ('\\.', Punctuation), ("[A-Z][\\w\\']*(?=\\s*\\.)", Name.Namespace), ("[A-Z][\\w\\']*", Name.Class, '#pop'), ("[a-z_][\\w\\']*", Name, '#pop'), default('#pop')]}