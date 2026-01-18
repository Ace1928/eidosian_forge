from pygments.lexer import RegexLexer, include, words
from pygments.token import Operator, Name, \
class TlbLexer(RegexLexer):
    """
    For TL-b source code.
    """
    name = 'Tl-b'
    aliases = ['tlb']
    filenames = ['*.tlb']
    tokens = {'root': [('\\s+', Whitespace), include('comments'), ('[0-9]+', Number), (words(('+', '-', '*', '=', '?', '~', '.', '^', '==', '<', '>', '<=', '>=', '!=')), Operator), (words(('##', '#<', '#<=')), Name.Tag), ('#[0-9a-f]*_?', Name.Tag), ('\\$[01]*_?', Name.Tag), ('[a-zA-Z_][0-9a-zA-Z_]*', Name), ('[;():\\[\\]{}]', Punctuation)], 'comments': [('//.*', Comment.Singleline), ('/\\*', Comment.Multiline, 'comment')], 'comment': [('[^/*]+', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)]}