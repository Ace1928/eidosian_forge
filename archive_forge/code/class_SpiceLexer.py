from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class SpiceLexer(RegexLexer):
    """
    For Spice source.

    .. versionadded:: 2.11
    """
    name = 'Spice'
    url = 'https://www.spicelang.com'
    filenames = ['*.spice']
    aliases = ['spice', 'spicelang']
    mimetypes = ['text/x-spice']
    tokens = {'root': [('\\n', Whitespace), ('\\s+', Whitespace), ('\\\\\\n', Text), ('//(.*?)\\n', Comment.Single), ('/(\\\\\\n)?[*]{2}(.|\\n)*?[*](\\\\\\n)?/', String.Doc), ('/(\\\\\\n)?[*](.|\\n)*?[*](\\\\\\n)?/', Comment.Multiline), ('(import|as)\\b', Keyword.Namespace), ('(f|p|type|struct|interface|enum|alias|operator)\\b', Keyword.Declaration), (words(('if', 'else', 'for', 'foreach', 'do', 'while', 'break', 'continue', 'return', 'assert', 'unsafe', 'ext'), suffix='\\b'), Keyword), (words(('const', 'signed', 'unsigned', 'inline', 'public', 'heap'), suffix='\\b'), Keyword.Pseudo), (words(('new', 'switch', 'case', 'yield', 'stash', 'pick', 'sync', 'class'), suffix='\\b'), Keyword.Reserved), ('(true|false|nil)\\b', Keyword.Constant), (words(('double', 'int', 'short', 'long', 'byte', 'char', 'string', 'bool', 'dyn'), suffix='\\b'), Keyword.Type), (words(('printf', 'sizeof', 'alignof', 'len'), suffix='\\b(\\()'), bygroups(Name.Builtin, Punctuation)), ('[-]?[0-9]*[.][0-9]+([eE][+-]?[0-9]+)?', Number.Double), ('0[bB][01]+[slu]?', Number.Bin), ('0[oO][0-7]+[slu]?', Number.Oct), ('0[xXhH][0-9a-fA-F]+[slu]?', Number.Hex), ('(0[dD])?[0-9]+[slu]?', Number.Integer), ('"(\\\\\\\\|\\\\[^\\\\]|[^"\\\\])*"', String), ("\\'(\\\\\\\\|\\\\[^\\\\]|[^\\'\\\\])\\'", String.Char), ('<<=|>>=|<<|>>|<=|>=|\\+=|-=|\\*=|/=|\\%=|\\|=|&=|\\^=|&&|\\|\\||&|\\||\\+\\+|--|\\%|\\^|\\~|==|!=|::|[.]{3}|#!|#|[+\\-*/&]', Operator), ('[|<>=!()\\[\\]{}.,;:\\?]', Punctuation), ('[^\\W\\d]\\w*', Name.Other)]}