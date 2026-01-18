import re
from pygments.lexer import RegexLexer, include, bygroups, using, words, \
from pygments.lexers.c_cpp import CppLexer, CLexer
from pygments.lexers.d import DLexer
from pygments.token import Text, Name, Number, String, Comment, Punctuation, \
class GasLexer(RegexLexer):
    """
    For Gas (AT&T) assembly code.
    """
    name = 'GAS'
    aliases = ['gas', 'asm']
    filenames = ['*.s', '*.S']
    mimetypes = ['text/x-gas']
    string = '"(\\\\"|[^"])*"'
    char = '[\\w$.@-]'
    identifier = '(?:[a-zA-Z$_]' + char + '*|\\.' + char + '+)'
    number = '(?:0[xX][a-zA-Z0-9]+|\\d+)'
    tokens = {'root': [include('whitespace'), (identifier + ':', Name.Label), ('\\.' + identifier, Name.Attribute, 'directive-args'), ('lock|rep(n?z)?|data\\d+', Name.Attribute), (identifier, Name.Function, 'instruction-args'), ('[\\r\\n]+', Text)], 'directive-args': [(identifier, Name.Constant), (string, String), ('@' + identifier, Name.Attribute), (number, Number.Integer), ('[\\r\\n]+', Text, '#pop'), include('punctuation'), include('whitespace')], 'instruction-args': [('([a-z0-9]+)( )(<)(' + identifier + ')(>)', bygroups(Number.Hex, Text, Punctuation, Name.Constant, Punctuation)), ('([a-z0-9]+)( )(<)(' + identifier + ')([-+])(' + number + ')(>)', bygroups(Number.Hex, Text, Punctuation, Name.Constant, Punctuation, Number.Integer, Punctuation)), (identifier, Name.Constant), (number, Number.Integer), ('%' + identifier, Name.Variable), ('$' + number, Number.Integer), ("$'(.|\\\\')'", String.Char), ('[\\r\\n]+', Text, '#pop'), include('punctuation'), include('whitespace')], 'whitespace': [('\\n', Text), ('\\s+', Text), ('[;#].*?\\n', Comment)], 'punctuation': [('[-*,.()\\[\\]!:]+', Punctuation)]}

    def analyse_text(text):
        if re.match('^\\.(text|data|section)', text, re.M):
            return True
        elif re.match('^\\.\\w+', text, re.M):
            return 0.1