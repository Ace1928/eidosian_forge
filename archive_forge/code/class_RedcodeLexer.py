from pygments.lexer import RegexLexer, include, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class RedcodeLexer(RegexLexer):
    """
    A simple Redcode lexer based on ICWS'94.
    Contributed by Adam Blinkinsop <blinks@acm.org>.

    .. versionadded:: 0.8
    """
    name = 'Redcode'
    aliases = ['redcode']
    filenames = ['*.cw']
    opcodes = ('DAT', 'MOV', 'ADD', 'SUB', 'MUL', 'DIV', 'MOD', 'JMP', 'JMZ', 'JMN', 'DJN', 'CMP', 'SLT', 'SPL', 'ORG', 'EQU', 'END')
    modifiers = ('A', 'B', 'AB', 'BA', 'F', 'X', 'I')
    tokens = {'root': [('\\s+', Text), (';.*$', Comment.Single), ('\\b(%s)\\b' % '|'.join(opcodes), Name.Function), ('\\b(%s)\\b' % '|'.join(modifiers), Name.Decorator), ('[A-Za-z_]\\w+', Name), ('[-+*/%]', Operator), ('[#$@<>]', Operator), ('[.,]', Punctuation), ('[-+]?\\d+', Number.Integer)]}