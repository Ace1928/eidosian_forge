import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
from pygments import unistring as uni
class JavaLexer(RegexLexer):
    """
    For `Java <http://www.sun.com/java/>`_ source code.
    """
    name = 'Java'
    aliases = ['java']
    filenames = ['*.java']
    mimetypes = ['text/x-java']
    flags = re.MULTILINE | re.DOTALL | re.UNICODE
    tokens = {'root': [('[^\\S\\n]+', Text), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline), ('(assert|break|case|catch|continue|default|do|else|finally|for|if|goto|instanceof|new|return|switch|this|throw|try|while)\\b', Keyword), ('((?:(?:[^\\W\\d]|\\$)[\\w.\\[\\]$<>]*\\s+)+?)((?:[^\\W\\d]|\\$)[\\w$]*)(\\s*)(\\()', bygroups(using(this), Name.Function, Text, Operator)), ('@[^\\W\\d][\\w.]*', Name.Decorator), ('(abstract|const|enum|extends|final|implements|native|private|protected|public|static|strictfp|super|synchronized|throws|transient|volatile)\\b', Keyword.Declaration), ('(boolean|byte|char|double|float|int|long|short|void)\\b', Keyword.Type), ('(package)(\\s+)', bygroups(Keyword.Namespace, Text), 'import'), ('(true|false|null)\\b', Keyword.Constant), ('(class|interface)(\\s+)', bygroups(Keyword.Declaration, Text), 'class'), ('(import(?:\\s+static)?)(\\s+)', bygroups(Keyword.Namespace, Text), 'import'), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ("'\\\\.'|'[^\\\\]'|'\\\\u[0-9a-fA-F]{4}'", String.Char), ('(\\.)((?:[^\\W\\d]|\\$)[\\w$]*)', bygroups(Operator, Name.Attribute)), ('^\\s*([^\\W\\d]|\\$)[\\w$]*:', Name.Label), ('([^\\W\\d]|\\$)[\\w$]*', Name), ('([0-9][0-9_]*\\.([0-9][0-9_]*)?|\\.[0-9][0-9_]*)([eE][+\\-]?[0-9][0-9_]*)?[fFdD]?|[0-9][eE][+\\-]?[0-9][0-9_]*[fFdD]?|[0-9]([eE][+\\-]?[0-9][0-9_]*)?[fFdD]|0[xX]([0-9a-fA-F][0-9a-fA-F_]*\\.?|([0-9a-fA-F][0-9a-fA-F_]*)?\\.[0-9a-fA-F][0-9a-fA-F_]*)[pP][+\\-]?[0-9][0-9_]*[fFdD]?', Number.Float), ('0[xX][0-9a-fA-F][0-9a-fA-F_]*[lL]?', Number.Hex), ('0[bB][01][01_]*[lL]?', Number.Bin), ('0[0-7_]+[lL]?', Number.Oct), ('0|[1-9][0-9_]*[lL]?', Number.Integer), ('[~^*!%&\\[\\](){}<>|+=:;,./?-]', Operator), ('\\n', Text)], 'class': [('([^\\W\\d]|\\$)[\\w$]*', Name.Class, '#pop')], 'import': [('[\\w.]+\\*?', Name.Namespace, '#pop')]}