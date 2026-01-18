import re
from pygments.lexer import RegexLexer, bygroups, default, words, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class BlitzBasicLexer(RegexLexer):
    """
    For `BlitzBasic <http://blitzbasic.com>`_ source code.

    .. versionadded:: 2.0
    """
    name = 'BlitzBasic'
    aliases = ['blitzbasic', 'b3d', 'bplus']
    filenames = ['*.bb', '*.decls']
    mimetypes = ['text/x-bb']
    bb_sktypes = '@{1,2}|[#$%]'
    bb_name = '[a-z]\\w*'
    bb_var = '(%s)(?:([ \\t]*)(%s)|([ \\t]*)([.])([ \\t]*)(?:(%s)))?' % (bb_name, bb_sktypes, bb_name)
    flags = re.MULTILINE | re.IGNORECASE
    tokens = {'root': [('[ \\t]+', Text), (';.*?\\n', Comment.Single), ('"', String.Double, 'string'), ('[0-9]+\\.[0-9]*(?!\\.)', Number.Float), ('\\.[0-9]+(?!\\.)', Number.Float), ('[0-9]+', Number.Integer), ('\\$[0-9a-f]+', Number.Hex), ('\\%[10]+', Number.Bin), (words(('Shl', 'Shr', 'Sar', 'Mod', 'Or', 'And', 'Not', 'Abs', 'Sgn', 'Handle', 'Int', 'Float', 'Str', 'First', 'Last', 'Before', 'After'), prefix='\\b', suffix='\\b'), Operator), ('([+\\-*/~=<>^])', Operator), ('[(),:\\[\\]\\\\]', Punctuation), ('\\.([ \\t]*)(%s)' % bb_name, Name.Label), ('\\b(New)\\b([ \\t]+)(%s)' % bb_name, bygroups(Keyword.Reserved, Text, Name.Class)), ('\\b(Gosub|Goto)\\b([ \\t]+)(%s)' % bb_name, bygroups(Keyword.Reserved, Text, Name.Label)), ('\\b(Object)\\b([ \\t]*)([.])([ \\t]*)(%s)\\b' % bb_name, bygroups(Operator, Text, Punctuation, Text, Name.Class)), ('\\b%s\\b([ \\t]*)(\\()' % bb_var, bygroups(Name.Function, Text, Keyword.Type, Text, Punctuation, Text, Name.Class, Text, Punctuation)), ('\\b(Function)\\b([ \\t]+)%s' % bb_var, bygroups(Keyword.Reserved, Text, Name.Function, Text, Keyword.Type, Text, Punctuation, Text, Name.Class)), ('\\b(Type)([ \\t]+)(%s)' % bb_name, bygroups(Keyword.Reserved, Text, Name.Class)), ('\\b(Pi|True|False|Null)\\b', Keyword.Constant), ('\\b(Local|Global|Const|Field|Dim)\\b', Keyword.Declaration), (words(('End', 'Return', 'Exit', 'Chr', 'Len', 'Asc', 'New', 'Delete', 'Insert', 'Include', 'Function', 'Type', 'If', 'Then', 'Else', 'ElseIf', 'EndIf', 'For', 'To', 'Next', 'Step', 'Each', 'While', 'Wend', 'Repeat', 'Until', 'Forever', 'Select', 'Case', 'Default', 'Goto', 'Gosub', 'Data', 'Read', 'Restore'), prefix='\\b', suffix='\\b'), Keyword.Reserved), (bb_var, bygroups(Name.Variable, Text, Keyword.Type, Text, Punctuation, Text, Name.Class))], 'string': [('""', String.Double), ('"C?', String.Double, '#pop'), ('[^"]+', String.Double)]}