import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
from pygments import unistring as uni
class TeaLangLexer(RegexLexer):
    """
    For `Tea <http://teatrove.org/>`_ source code. Only used within a
    TeaTemplateLexer.

    .. versionadded:: 1.5
    """
    flags = re.MULTILINE | re.DOTALL
    tokens = {'root': [('^(\\s*(?:[a-zA-Z_][\\w\\.\\[\\]]*\\s+)+?)([a-zA-Z_]\\w*)(\\s*)(\\()', bygroups(using(this), Name.Function, Text, Operator)), ('[^\\S\\n]+', Text), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline), ('@[a-zA-Z_][\\w\\.]*', Name.Decorator), ('(and|break|else|foreach|if|in|not|or|reverse)\\b', Keyword), ('(as|call|define)\\b', Keyword.Declaration), ('(true|false|null)\\b', Keyword.Constant), ('(template)(\\s+)', bygroups(Keyword.Declaration, Text), 'template'), ('(import)(\\s+)', bygroups(Keyword.Namespace, Text), 'import'), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ("\\'(\\\\\\\\|\\\\\\'|[^\\'])*\\'", String), ('(\\.)([a-zA-Z_]\\w*)', bygroups(Operator, Name.Attribute)), ('[a-zA-Z_]\\w*:', Name.Label), ('[a-zA-Z_\\$]\\w*', Name), ('(isa|[.]{3}|[.]{2}|[=#!<>+-/%&;,.\\*\\\\\\(\\)\\[\\]\\{\\}])', Operator), ('[0-9][0-9]*\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('0x[0-9a-fA-F]+', Number.Hex), ('[0-9]+L?', Number.Integer), ('\\n', Text)], 'template': [('[a-zA-Z_]\\w*', Name.Class, '#pop')], 'import': [('[\\w.]+\\*?', Name.Namespace, '#pop')]}