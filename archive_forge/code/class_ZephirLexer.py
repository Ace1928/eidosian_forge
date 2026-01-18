import re
from pygments.lexer import RegexLexer, include, bygroups, default, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, get_list_opt, iteritems
class ZephirLexer(RegexLexer):
    """
    For `Zephir language <http://zephir-lang.com/>`_ source code.

    Zephir is a compiled high level language aimed
    to the creation of C-extensions for PHP.

    .. versionadded:: 2.0
    """
    name = 'Zephir'
    aliases = ['zephir']
    filenames = ['*.zep']
    zephir_keywords = ['fetch', 'echo', 'isset', 'empty']
    zephir_type = ['bit', 'bits', 'string']
    flags = re.DOTALL | re.MULTILINE
    tokens = {'commentsandwhitespace': [('\\s+', Text), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline)], 'slashstartsregex': [include('commentsandwhitespace'), ('/(\\\\.|[^[/\\\\\\n]|\\[(\\\\.|[^\\]\\\\\\n])*])+/([gim]+\\b|\\B)', String.Regex, '#pop'), default('#pop')], 'badregex': [('\\n', Text, '#pop')], 'root': [('^(?=\\s|/|<!--)', Text, 'slashstartsregex'), include('commentsandwhitespace'), ('\\+\\+|--|~|&&|\\?|:|\\|\\||\\\\(?=\\n)|(<<|>>>?|==?|!=?|->|[-<>+*%&|^/])=?', Operator, 'slashstartsregex'), ('[{(\\[;,]', Punctuation, 'slashstartsregex'), ('[})\\].]', Punctuation), ('(for|in|while|do|break|return|continue|switch|case|default|if|else|loop|require|inline|throw|try|catch|finally|new|delete|typeof|instanceof|void|namespace|use|extends|this|fetch|isset|unset|echo|fetch|likely|unlikely|empty)\\b', Keyword, 'slashstartsregex'), ('(var|let|with|function)\\b', Keyword.Declaration, 'slashstartsregex'), ('(abstract|boolean|bool|char|class|const|double|enum|export|extends|final|native|goto|implements|import|int|string|interface|long|ulong|char|uchar|float|unsigned|private|protected|public|short|static|self|throws|reverse|transient|volatile)\\b', Keyword.Reserved), ('(true|false|null|undefined)\\b', Keyword.Constant), ('(Array|Boolean|Date|_REQUEST|_COOKIE|_SESSION|_GET|_POST|_SERVER|this|stdClass|range|count|iterator|window)\\b', Name.Builtin), ('[$a-zA-Z_][\\w\\\\]*', Name.Other), ('[0-9][0-9]*\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('0x[0-9a-fA-F]+', Number.Hex), ('[0-9]+', Number.Integer), ('"(\\\\\\\\|\\\\"|[^"])*"', String.Double), ("'(\\\\\\\\|\\\\'|[^'])*'", String.Single)]}