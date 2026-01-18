import re
from pygments.lexer import RegexLexer, bygroups, default, words, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class BlitzMaxLexer(RegexLexer):
    """
    For `BlitzMax <http://blitzbasic.com>`_ source code.

    .. versionadded:: 1.4
    """
    name = 'BlitzMax'
    aliases = ['blitzmax', 'bmax']
    filenames = ['*.bmx']
    mimetypes = ['text/x-bmx']
    bmax_vopwords = '\\b(Shl|Shr|Sar|Mod)\\b'
    bmax_sktypes = '@{1,2}|[!#$%]'
    bmax_lktypes = '\\b(Int|Byte|Short|Float|Double|Long)\\b'
    bmax_name = '[a-z_]\\w*'
    bmax_var = '(%s)(?:(?:([ \\t]*)(%s)|([ \\t]*:[ \\t]*\\b(?:Shl|Shr|Sar|Mod)\\b)|([ \\t]*)(:)([ \\t]*)(?:%s|(%s)))(?:([ \\t]*)(Ptr))?)' % (bmax_name, bmax_sktypes, bmax_lktypes, bmax_name)
    bmax_func = bmax_var + '?((?:[ \\t]|\\.\\.\\n)*)([(])'
    flags = re.MULTILINE | re.IGNORECASE
    tokens = {'root': [('[ \\t]+', Text), ('\\.\\.\\n', Text), ("'.*?\\n", Comment.Single), ('([ \\t]*)\\bRem\\n(\\n|.)*?\\s*\\bEnd([ \\t]*)Rem', Comment.Multiline), ('"', String.Double, 'string'), ('[0-9]+\\.[0-9]*(?!\\.)', Number.Float), ('\\.[0-9]*(?!\\.)', Number.Float), ('[0-9]+', Number.Integer), ('\\$[0-9a-f]+', Number.Hex), ('\\%[10]+', Number.Bin), ('(?:(?:(:)?([ \\t]*)(:?%s|([+\\-*/&|~]))|Or|And|Not|[=<>^]))' % bmax_vopwords, Operator), ('[(),.:\\[\\]]', Punctuation), ('(?:#[\\w \\t]*)', Name.Label), ('(?:\\?[\\w \\t]*)', Comment.Preproc), ('\\b(New)\\b([ \\t]?)([(]?)(%s)' % bmax_name, bygroups(Keyword.Reserved, Text, Punctuation, Name.Class)), ('\\b(Import|Framework|Module)([ \\t]+)(%s\\.%s)' % (bmax_name, bmax_name), bygroups(Keyword.Reserved, Text, Keyword.Namespace)), (bmax_func, bygroups(Name.Function, Text, Keyword.Type, Operator, Text, Punctuation, Text, Keyword.Type, Name.Class, Text, Keyword.Type, Text, Punctuation)), (bmax_var, bygroups(Name.Variable, Text, Keyword.Type, Operator, Text, Punctuation, Text, Keyword.Type, Name.Class, Text, Keyword.Type)), ('\\b(Type|Extends)([ \\t]+)(%s)' % bmax_name, bygroups(Keyword.Reserved, Text, Name.Class)), ('\\b(Ptr)\\b', Keyword.Type), ('\\b(Pi|True|False|Null|Self|Super)\\b', Keyword.Constant), ('\\b(Local|Global|Const|Field)\\b', Keyword.Declaration), (words(('TNullMethodException', 'TNullFunctionException', 'TNullObjectException', 'TArrayBoundsException', 'TRuntimeException'), prefix='\\b', suffix='\\b'), Name.Exception), (words(('Strict', 'SuperStrict', 'Module', 'ModuleInfo', 'End', 'Return', 'Continue', 'Exit', 'Public', 'Private', 'Var', 'VarPtr', 'Chr', 'Len', 'Asc', 'SizeOf', 'Sgn', 'Abs', 'Min', 'Max', 'New', 'Release', 'Delete', 'Incbin', 'IncbinPtr', 'IncbinLen', 'Framework', 'Include', 'Import', 'Extern', 'EndExtern', 'Function', 'EndFunction', 'Type', 'EndType', 'Extends', 'Method', 'EndMethod', 'Abstract', 'Final', 'If', 'Then', 'Else', 'ElseIf', 'EndIf', 'For', 'To', 'Next', 'Step', 'EachIn', 'While', 'Wend', 'EndWhile', 'Repeat', 'Until', 'Forever', 'Select', 'Case', 'Default', 'EndSelect', 'Try', 'Catch', 'EndTry', 'Throw', 'Assert', 'Goto', 'DefData', 'ReadData', 'RestoreData'), prefix='\\b', suffix='\\b'), Keyword.Reserved), ('(%s)' % bmax_name, Name.Variable)], 'string': [('""', String.Double), ('"C?', String.Double, '#pop'), ('[^"]+', String.Double)]}