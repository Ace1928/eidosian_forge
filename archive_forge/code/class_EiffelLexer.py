from pygments.lexer import RegexLexer, include, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class EiffelLexer(RegexLexer):
    """
    For `Eiffel <http://www.eiffel.com>`_ source code.

    .. versionadded:: 2.0
    """
    name = 'Eiffel'
    aliases = ['eiffel']
    filenames = ['*.e']
    mimetypes = ['text/x-eiffel']
    tokens = {'root': [('[^\\S\\n]+', Text), ('--.*?\\n', Comment.Single), ('[^\\S\\n]+', Text), ('(?i)(true|false|void|current|result|precursor)\\b', Keyword.Constant), ('(?i)(and(\\s+then)?|not|xor|implies|or(\\s+else)?)\\b', Operator.Word), (words(('across', 'agent', 'alias', 'all', 'as', 'assign', 'attached', 'attribute', 'check', 'class', 'convert', 'create', 'debug', 'deferred', 'detachable', 'do', 'else', 'elseif', 'end', 'ensure', 'expanded', 'export', 'external', 'feature', 'from', 'frozen', 'if', 'inherit', 'inspect', 'invariant', 'like', 'local', 'loop', 'none', 'note', 'obsolete', 'old', 'once', 'only', 'redefine', 'rename', 'require', 'rescue', 'retry', 'select', 'separate', 'then', 'undefine', 'until', 'variant', 'when'), prefix='(?i)\\b', suffix='\\b'), Keyword.Reserved), ('"\\[(([^\\]%]|\\n)|%(.|\\n)|\\][^"])*?\\]"', String), ('"([^"%\\n]|%.)*?"', String), include('numbers'), ("'([^'%]|%'|%%)'", String.Char), ('(//|\\\\\\\\|>=|<=|:=|/=|~|/~|[\\\\?!#%&@|+/\\-=>*$<^\\[\\]])', Operator), ('([{}():;,.])', Punctuation), ('([a-z]\\w*)|([A-Z][A-Z0-9_]*[a-z]\\w*)', Name), ('([A-Z][A-Z0-9_]*)', Name.Class), ('\\n+', Text)], 'numbers': [('0[xX][a-fA-F0-9]+', Number.Hex), ('0[bB][01]+', Number.Bin), ('0[cC][0-7]+', Number.Oct), ('([0-9]+\\.[0-9]*)|([0-9]*\\.[0-9]+)', Number.Float), ('[0-9]+', Number.Integer)]}