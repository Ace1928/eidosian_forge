from pygments.lexer import RegexLexer, words
from pygments.token import Name, Text, Keyword, Whitespace, Number, Comment
import re
class AMDGPULexer(RegexLexer):
    """
    For AMD GPU assembly.

    .. versionadded:: 2.8
    """
    name = 'AMDGPU'
    aliases = ['amdgpu']
    filenames = ['*.isa']
    flags = re.IGNORECASE
    tokens = {'root': [('\\s+', Whitespace), ('[\\r\\n]+', Text), ('(([a-z_0-9])*:([a-z_0-9])*)', Name.Attribute), ('(\\[|\\]|\\(|\\)|,|\\:|\\&)', Text), ('([;#]|//).*?\\n', Comment.Single), ('((s_)?(scratch|ds|buffer|flat|image)_[a-z0-9_]+)', Keyword.Reserved), ('(_lo|_hi)', Name.Variable), ('(vmcnt|lgkmcnt|expcnt)', Name.Attribute), ('(attr[0-9].[a-z])', Name.Attribute), (words(('op', 'vaddr', 'vdata', 'off', 'soffset', 'srsrc', 'format', 'offset', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe', 'lds', 'lit', 'unorm'), suffix='\\b'), Name.Attribute), ('(label_[a-z0-9]+)', Keyword), ('(_L[0-9]*)', Name.Variable), ('(s|v)_[a-z0-9_]+', Keyword), ('(v[0-9.]+|vcc|exec|v)', Name.Variable), ('s[0-9.]+|s', Name.Variable), ('[0-9]+\\.[^0-9]+', Number.Float), ('(0[xX][a-z0-9]+)|([0-9]+)', Number.Integer)]}