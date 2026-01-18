import re
from pygments.lexer import RegexLexer, include, bygroups, using, words, \
from pygments.lexers.c_cpp import CppLexer, CLexer
from pygments.lexers.d import DLexer
from pygments.token import Text, Name, Number, String, Comment, Punctuation, \
class Ca65Lexer(RegexLexer):
    """
    For ca65 assembler sources.

    .. versionadded:: 1.6
    """
    name = 'ca65 assembler'
    aliases = ['ca65']
    filenames = ['*.s']
    flags = re.IGNORECASE
    tokens = {'root': [(';.*', Comment.Single), ('\\s+', Text), ('[a-z_.@$][\\w.@$]*:', Name.Label), ('((ld|st)[axy]|(in|de)[cxy]|asl|lsr|ro[lr]|adc|sbc|cmp|cp[xy]|cl[cvdi]|se[cdi]|jmp|jsr|bne|beq|bpl|bmi|bvc|bvs|bcc|bcs|p[lh][ap]|rt[is]|brk|nop|ta[xy]|t[xy]a|txs|tsx|and|ora|eor|bit)\\b', Keyword), ('\\.\\w+', Keyword.Pseudo), ('[-+~*/^&|!<>=]', Operator), ('"[^"\\n]*.', String), ("'[^'\\n]*.", String.Char), ('\\$[0-9a-f]+|[0-9a-f]+h\\b', Number.Hex), ('\\d+', Number.Integer), ('%[01]+', Number.Bin), ('[#,.:()=\\[\\]]', Punctuation), ('[a-z_.@$][\\w.@$]*', Name)]}

    def analyse_text(self, text):
        if re.match('^\\s*;', text, re.MULTILINE):
            return 0.9