import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class BaseMakefileLexer(RegexLexer):
    """
    Lexer for simple Makefiles (no preprocessing).

    .. versionadded:: 0.10
    """
    name = 'Base Makefile'
    aliases = ['basemake']
    filenames = []
    mimetypes = []
    tokens = {'root': [('^(?:[\\t ]+.*\\n|\\n)+', using(BashLexer)), ('\\$[<@$+%?|*]', Keyword), ('\\s+', Text), ('#.*?\\n', Comment), ('(export)(\\s+)(?=[\\w${}\\t -]+\\n)', bygroups(Keyword, Text), 'export'), ('export\\s+', Keyword), ('([\\w${}().-]+)(\\s*)([!?:+]?=)([ \\t]*)((?:.*\\\\\\n)+|.*\\n)', bygroups(Name.Variable, Text, Operator, Text, using(BashLexer))), ('(?s)"(\\\\\\\\|\\\\.|[^"\\\\])*"', String.Double), ("(?s)'(\\\\\\\\|\\\\.|[^'\\\\])*'", String.Single), ('([^\\n:]+)(:+)([ \\t]*)', bygroups(Name.Function, Operator, Text), 'block-header'), ('\\$\\(', Keyword, 'expansion')], 'expansion': [('[^$a-zA-Z_()]+', Text), ('[a-zA-Z_]+', Name.Variable), ('\\$', Keyword), ('\\(', Keyword, '#push'), ('\\)', Keyword, '#pop')], 'export': [('[\\w${}-]+', Name.Variable), ('\\n', Text, '#pop'), ('\\s+', Text)], 'block-header': [('[,|]', Punctuation), ('#.*?\\n', Comment, '#pop'), ('\\\\\\n', Text), ('\\$\\(', Keyword, 'expansion'), ('[a-zA-Z_]+', Name), ('\\n', Text, '#pop'), ('.', Text)]}