import re
from pygments.lexer import RegexLexer, include, bygroups, using, words, \
from pygments.lexers.c_cpp import CppLexer, CLexer
from pygments.lexers.d import DLexer
from pygments.token import Text, Name, Number, String, Comment, Punctuation, \
class Dasm16Lexer(RegexLexer):
    """
    For DCPU-16 Assembly.

    .. versionadded:: 2.4
    """
    name = 'DASM16'
    url = 'http://0x10c.com/doc/dcpu-16.txt'
    aliases = ['dasm16']
    filenames = ['*.dasm16', '*.dasm']
    mimetypes = ['text/x-dasm16']
    INSTRUCTIONS = ['SET', 'ADD', 'SUB', 'MUL', 'MLI', 'DIV', 'DVI', 'MOD', 'MDI', 'AND', 'BOR', 'XOR', 'SHR', 'ASR', 'SHL', 'IFB', 'IFC', 'IFE', 'IFN', 'IFG', 'IFA', 'IFL', 'IFU', 'ADX', 'SBX', 'STI', 'STD', 'JSR', 'INT', 'IAG', 'IAS', 'RFI', 'IAQ', 'HWN', 'HWQ', 'HWI']
    REGISTERS = ['A', 'B', 'C', 'X', 'Y', 'Z', 'I', 'J', 'SP', 'PC', 'EX', 'POP', 'PEEK', 'PUSH']
    char = '[a-zA-Z0-9_$@.]'
    identifier = '(?:[a-zA-Z$_]' + char + '*|\\.' + char + '+)'
    number = '[+-]?(?:0[xX][a-zA-Z0-9]+|\\d+)'
    binary_number = '0b[01_]+'
    instruction = '(?i)(' + '|'.join(INSTRUCTIONS) + ')'
    single_char = "'\\\\?" + char + "'"
    string = '"(\\\\"|[^"])*"'

    def guess_identifier(lexer, match):
        ident = match.group(0)
        klass = Name.Variable if ident.upper() in lexer.REGISTERS else Name.Label
        yield (match.start(), klass, ident)
    tokens = {'root': [include('whitespace'), (':' + identifier, Name.Label), (identifier + ':', Name.Label), (instruction, Name.Function, 'instruction-args'), ('\\.' + identifier, Name.Function, 'data-args'), ('[\\r\\n]+', Whitespace)], 'numeric': [(binary_number, Number.Integer), (number, Number.Integer), (single_char, String)], 'arg': [(identifier, guess_identifier), include('numeric')], 'deref': [('\\+', Punctuation), ('\\]', Punctuation, '#pop'), include('arg'), include('whitespace')], 'instruction-line': [('[\\r\\n]+', Whitespace, '#pop'), (';.*?$', Comment, '#pop'), include('whitespace')], 'instruction-args': [(',', Punctuation), ('\\[', Punctuation, 'deref'), include('arg'), include('instruction-line')], 'data-args': [(',', Punctuation), include('numeric'), (string, String), include('instruction-line')], 'whitespace': [('\\n', Whitespace), ('\\s+', Whitespace), (';.*?\\n', Comment)]}