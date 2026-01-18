import re
from pygments.lexer import RegexLexer, include, bygroups, default, combined, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, get_list_opt, iteritems
class MoonScriptLexer(LuaLexer):
    """
    For `MoonScript <http://moonscript.org>`_ source code.

    .. versionadded:: 1.5
    """
    name = 'MoonScript'
    aliases = ['moon', 'moonscript']
    filenames = ['*.moon']
    mimetypes = ['text/x-moonscript', 'application/x-moonscript']
    tokens = {'root': [('#!(.*?)$', Comment.Preproc), default('base')], 'base': [('--.*$', Comment.Single), ('(?i)(\\d*\\.\\d+|\\d+\\.\\d*)(e[+-]?\\d+)?', Number.Float), ('(?i)\\d+e[+-]?\\d+', Number.Float), ('(?i)0x[0-9a-f]*', Number.Hex), ('\\d+', Number.Integer), ('\\n', Text), ('[^\\S\\n]+', Text), ('(?s)\\[(=*)\\[.*?\\]\\1\\]', String), ('(->|=>)', Name.Function), (':[a-zA-Z_]\\w*', Name.Variable), ('(==|!=|~=|<=|>=|\\.\\.\\.|\\.\\.|[=+\\-*/%^<>#!.\\\\:])', Operator), ('[;,]', Punctuation), ('[\\[\\]{}()]', Keyword.Type), ('[a-zA-Z_]\\w*:', Name.Variable), (words(('class', 'extends', 'if', 'then', 'super', 'do', 'with', 'import', 'export', 'while', 'elseif', 'return', 'for', 'in', 'from', 'when', 'using', 'else', 'and', 'or', 'not', 'switch', 'break'), suffix='\\b'), Keyword), ('(true|false|nil)\\b', Keyword.Constant), ('(and|or|not)\\b', Operator.Word), ('(self)\\b', Name.Builtin.Pseudo), ('@@?([a-zA-Z_]\\w*)?', Name.Variable.Class), ('[A-Z]\\w*', Name.Class), ('[A-Za-z_]\\w*(\\.[A-Za-z_]\\w*)?', Name), ("'", String.Single, combined('stringescape', 'sqs')), ('"', String.Double, combined('stringescape', 'dqs'))], 'stringescape': [('\\\\([abfnrtv\\\\"\']|\\d{1,3})', String.Escape)], 'sqs': [("'", String.Single, '#pop'), ('.', String)], 'dqs': [('"', String.Double, '#pop'), ('.', String)]}

    def get_tokens_unprocessed(self, text):
        for index, token, value in LuaLexer.get_tokens_unprocessed(self, text):
            if token == Punctuation and value == '.':
                token = Operator
            yield (index, token, value)