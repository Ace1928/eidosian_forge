import re
from pygments.lexer import RegexLexer, include, bygroups, using, words, \
from pygments.lexers.c_cpp import CppLexer, CLexer
from pygments.lexers.d import DLexer
from pygments.token import Text, Name, Number, String, Comment, Punctuation, \
class NasmLexer(RegexLexer):
    """
    For Nasm (Intel) assembly code.
    """
    name = 'NASM'
    aliases = ['nasm']
    filenames = ['*.asm', '*.ASM']
    mimetypes = ['text/x-nasm']
    identifier = '[a-z$._?][\\w$.?#@~]*'
    hexn = '(?:0x[0-9a-f]+|$0[0-9a-f]*|[0-9]+[0-9a-f]*h)'
    octn = '[0-7]+q'
    binn = '[01]+b'
    decn = '[0-9]+'
    floatn = decn + '\\.e?' + decn
    string = '"(\\\\"|[^"\\n])*"|' + "'(\\\\'|[^'\\n])*'|" + '`(\\\\`|[^`\\n])*`'
    declkw = '(?:res|d)[bwdqt]|times'
    register = 'r[0-9][0-5]?[bwd]|[a-d][lh]|[er]?[a-d]x|[er]?[sb]p|[er]?[sd]i|[c-gs]s|st[0-7]|mm[0-7]|cr[0-4]|dr[0-367]|tr[3-7]'
    wordop = 'seg|wrt|strict'
    type = 'byte|[dq]?word'
    directives = 'BITS|USE16|USE32|SECTION|SEGMENT|ABSOLUTE|EXTERN|GLOBAL|ORG|ALIGN|STRUC|ENDSTRUC|COMMON|CPU|GROUP|UPPERCASE|IMPORT|EXPORT|LIBRARY|MODULE'
    flags = re.IGNORECASE | re.MULTILINE
    tokens = {'root': [('^\\s*%', Comment.Preproc, 'preproc'), include('whitespace'), (identifier + ':', Name.Label), ('(%s)(\\s+)(equ)' % identifier, bygroups(Name.Constant, Keyword.Declaration, Keyword.Declaration), 'instruction-args'), (directives, Keyword, 'instruction-args'), (declkw, Keyword.Declaration, 'instruction-args'), (identifier, Name.Function, 'instruction-args'), ('[\\r\\n]+', Text)], 'instruction-args': [(string, String), (hexn, Number.Hex), (octn, Number.Oct), (binn, Number.Bin), (floatn, Number.Float), (decn, Number.Integer), include('punctuation'), (register, Name.Builtin), (identifier, Name.Variable), ('[\\r\\n]+', Text, '#pop'), include('whitespace')], 'preproc': [('[^;\\n]+', Comment.Preproc), (';.*?\\n', Comment.Single, '#pop'), ('\\n', Comment.Preproc, '#pop')], 'whitespace': [('\\n', Text), ('[ \\t]+', Text), (';.*', Comment.Single)], 'punctuation': [('[,():\\[\\]]+', Punctuation), ('[&|^<>+*/%~-]+', Operator), ('[$]+', Keyword.Constant), (wordop, Operator.Word), (type, Keyword.Type)]}