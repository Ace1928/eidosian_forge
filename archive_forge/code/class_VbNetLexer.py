import re
from pygments.lexer import RegexLexer, DelegatingLexer, bygroups, include, \
from pygments.token import Punctuation, \
from pygments.util import get_choice_opt, iteritems
from pygments import unistring as uni
from pygments.lexers.html import XmlLexer
class VbNetLexer(RegexLexer):
    """
    For
    `Visual Basic.NET <http://msdn2.microsoft.com/en-us/vbasic/default.aspx>`_
    source code.
    """
    name = 'VB.net'
    aliases = ['vb.net', 'vbnet']
    filenames = ['*.vb', '*.bas']
    mimetypes = ['text/x-vbnet', 'text/x-vba']
    uni_name = '[_' + uni.combine('Ll', 'Lt', 'Lm', 'Nl') + ']' + '[' + uni.combine('Ll', 'Lt', 'Lm', 'Nl', 'Nd', 'Pc', 'Cf', 'Mn', 'Mc') + ']*'
    flags = re.MULTILINE | re.IGNORECASE
    tokens = {'root': [('^\\s*<.*?>', Name.Attribute), ('\\s+', Text), ('\\n', Text), ('rem\\b.*?\\n', Comment), ("'.*?\\n", Comment), ('#If\\s.*?\\sThen|#ElseIf\\s.*?\\sThen|#Else|#End\\s+If|#Const|#ExternalSource.*?\\n|#End\\s+ExternalSource|#Region.*?\\n|#End\\s+Region|#ExternalChecksum', Comment.Preproc), ('[(){}!#,.:]', Punctuation), ('Option\\s+(Strict|Explicit|Compare)\\s+(On|Off|Binary|Text)', Keyword.Declaration), (words(('AddHandler', 'Alias', 'ByRef', 'ByVal', 'Call', 'Case', 'Catch', 'CBool', 'CByte', 'CChar', 'CDate', 'CDec', 'CDbl', 'CInt', 'CLng', 'CObj', 'Continue', 'CSByte', 'CShort', 'CSng', 'CStr', 'CType', 'CUInt', 'CULng', 'CUShort', 'Declare', 'Default', 'Delegate', 'DirectCast', 'Do', 'Each', 'Else', 'ElseIf', 'EndIf', 'Erase', 'Error', 'Event', 'Exit', 'False', 'Finally', 'For', 'Friend', 'Get', 'Global', 'GoSub', 'GoTo', 'Handles', 'If', 'Implements', 'Inherits', 'Interface', 'Let', 'Lib', 'Loop', 'Me', 'MustInherit', 'MustOverride', 'MyBase', 'MyClass', 'Narrowing', 'New', 'Next', 'Not', 'Nothing', 'NotInheritable', 'NotOverridable', 'Of', 'On', 'Operator', 'Option', 'Optional', 'Overloads', 'Overridable', 'Overrides', 'ParamArray', 'Partial', 'Private', 'Protected', 'Public', 'RaiseEvent', 'ReadOnly', 'ReDim', 'RemoveHandler', 'Resume', 'Return', 'Select', 'Set', 'Shadows', 'Shared', 'Single', 'Static', 'Step', 'Stop', 'SyncLock', 'Then', 'Throw', 'To', 'True', 'Try', 'TryCast', 'Wend', 'Using', 'When', 'While', 'Widening', 'With', 'WithEvents', 'WriteOnly'), prefix='(?<!\\.)', suffix='\\b'), Keyword), ('(?<!\\.)End\\b', Keyword, 'end'), ('(?<!\\.)(Dim|Const)\\b', Keyword, 'dim'), ('(?<!\\.)(Function|Sub|Property)(\\s+)', bygroups(Keyword, Text), 'funcname'), ('(?<!\\.)(Class|Structure|Enum)(\\s+)', bygroups(Keyword, Text), 'classname'), ('(?<!\\.)(Module|Namespace|Imports)(\\s+)', bygroups(Keyword, Text), 'namespace'), ('(?<!\\.)(Boolean|Byte|Char|Date|Decimal|Double|Integer|Long|Object|SByte|Short|Single|String|Variant|UInteger|ULong|UShort)\\b', Keyword.Type), ('(?<!\\.)(AddressOf|And|AndAlso|As|GetType|In|Is|IsNot|Like|Mod|Or|OrElse|TypeOf|Xor)\\b', Operator.Word), ('&=|[*]=|/=|\\\\=|\\^=|\\+=|-=|<<=|>>=|<<|>>|:=|<=|>=|<>|[-&*/\\\\^+=<>\\[\\]]', Operator), ('"', String, 'string'), ('_\\n', Text), (uni_name + '[%&@!#$]?', Name), ('#.*?#', Literal.Date), ('(\\d+\\.\\d*|\\d*\\.\\d+)(F[+-]?[0-9]+)?', Number.Float), ('\\d+([SILDFR]|US|UI|UL)?', Number.Integer), ('&H[0-9a-f]+([SILDFR]|US|UI|UL)?', Number.Integer), ('&O[0-7]+([SILDFR]|US|UI|UL)?', Number.Integer)], 'string': [('""', String), ('"C?', String, '#pop'), ('[^"]+', String)], 'dim': [(uni_name, Name.Variable, '#pop'), default('#pop')], 'funcname': [(uni_name, Name.Function, '#pop')], 'classname': [(uni_name, Name.Class, '#pop')], 'namespace': [(uni_name, Name.Namespace), ('\\.', Name.Namespace), default('#pop')], 'end': [('\\s+', Text), ('(Function|Sub|Property|Class|Structure|Enum|Module|Namespace)\\b', Keyword, '#pop'), default('#pop')]}

    def analyse_text(text):
        if re.search('^\\s*(#If|Module|Namespace)', text, re.MULTILINE):
            return 0.5