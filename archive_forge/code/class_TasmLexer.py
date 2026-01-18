import re
from pygments.lexer import RegexLexer, include, bygroups, using, words, \
from pygments.lexers.c_cpp import CppLexer, CLexer
from pygments.lexers.d import DLexer
from pygments.token import Text, Name, Number, String, Comment, Punctuation, \
class TasmLexer(RegexLexer):
    """
    For Tasm (Turbo Assembler) assembly code.
    """
    name = 'TASM'
    aliases = ['tasm']
    filenames = ['*.asm', '*.ASM', '*.tasm']
    mimetypes = ['text/x-tasm']
    identifier = '[@a-z$._?][\\w$.?#@~]*'
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
    directives = 'BITS|USE16|USE32|SECTION|SEGMENT|ABSOLUTE|EXTERN|GLOBAL|ORG|ALIGN|STRUC|ENDSTRUC|ENDS|COMMON|CPU|GROUP|UPPERCASE|INCLUDE|EXPORT|LIBRARY|MODULE|PROC|ENDP|USES|ARG|DATASEG|UDATASEG|END|IDEAL|P386|MODEL|ASSUME|CODESEG|SIZE'
    datatype = 'db|dd|dw|T[A-Z][a-z]+'
    flags = re.IGNORECASE | re.MULTILINE
    tokens = {'root': [('^\\s*%', Comment.Preproc, 'preproc'), include('whitespace'), (identifier + ':', Name.Label), (directives, Keyword, 'instruction-args'), ('(%s)(\\s+)(%s)' % (identifier, datatype), bygroups(Name.Constant, Keyword.Declaration, Keyword.Declaration), 'instruction-args'), (declkw, Keyword.Declaration, 'instruction-args'), (identifier, Name.Function, 'instruction-args'), ('[\\r\\n]+', Text)], 'instruction-args': [(string, String), (hexn, Number.Hex), (octn, Number.Oct), (binn, Number.Bin), (floatn, Number.Float), (decn, Number.Integer), include('punctuation'), (register, Name.Builtin), (identifier, Name.Variable), ('(\\\\\\s*)(;.*)([\\r\\n])', bygroups(Text, Comment.Single, Text)), ('[\\r\\n]+', Text, '#pop'), include('whitespace')], 'preproc': [('[^;\\n]+', Comment.Preproc), (';.*?\\n', Comment.Single, '#pop'), ('\\n', Comment.Preproc, '#pop')], 'whitespace': [('[\\n\\r]', Text), ('\\\\[\\n\\r]', Text), ('[ \\t]+', Text), (';.*', Comment.Single)], 'punctuation': [('[,():\\[\\]]+', Punctuation), ('[&|^<>+*=/%~-]+', Operator), ('[$]+', Keyword.Constant), (wordop, Operator.Word), (type, Keyword.Type)]}