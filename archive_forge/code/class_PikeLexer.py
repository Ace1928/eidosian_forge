import re
from pygments.lexer import RegexLexer, include, bygroups, inherit, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers import _mql_builtins
class PikeLexer(CppLexer):
    """
    For `Pike <http://pike.lysator.liu.se/>`_ source code.

    .. versionadded:: 2.0
    """
    name = 'Pike'
    aliases = ['pike']
    filenames = ['*.pike', '*.pmod']
    mimetypes = ['text/x-pike']
    tokens = {'statements': [(words(('catch', 'new', 'private', 'protected', 'public', 'gauge', 'throw', 'throws', 'class', 'interface', 'implement', 'abstract', 'extends', 'from', 'this', 'super', 'constant', 'final', 'static', 'import', 'use', 'extern', 'inline', 'proto', 'break', 'continue', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'as', 'in', 'version', 'return', 'true', 'false', 'null', '__VERSION__', '__MAJOR__', '__MINOR__', '__BUILD__', '__REAL_VERSION__', '__REAL_MAJOR__', '__REAL_MINOR__', '__REAL_BUILD__', '__DATE__', '__TIME__', '__FILE__', '__DIR__', '__LINE__', '__AUTO_BIGNUM__', '__NT__', '__PIKE__', '__amigaos__', '_Pragma', 'static_assert', 'defined', 'sscanf'), suffix='\\b'), Keyword), ('(bool|int|long|float|short|double|char|string|object|void|mapping|array|multiset|program|function|lambda|mixed|[a-z_][a-z0-9_]*_t)\\b', Keyword.Type), ('(class)(\\s+)', bygroups(Keyword, Text), 'classname'), ('[~!%^&*+=|?:<>/@-]', Operator), inherit], 'classname': [('[a-zA-Z_]\\w*', Name.Class, '#pop'), ('\\s*(?=>)', Text, '#pop')]}