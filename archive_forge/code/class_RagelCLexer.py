import re
from pygments.lexer import RegexLexer, DelegatingLexer, \
from pygments.token import Punctuation, Other, Text, Comment, Operator, \
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers.objective import ObjectiveCLexer
from pygments.lexers.d import DLexer
from pygments.lexers.dotnet import CSharpLexer
from pygments.lexers.ruby import RubyLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.perl import PerlLexer
class RagelCLexer(DelegatingLexer):
    """
    A lexer for `Ragel`_ in a C host file.

    .. versionadded:: 1.1
    """
    name = 'Ragel in C Host'
    aliases = ['ragel-c']
    filenames = ['*.rl']

    def __init__(self, **options):
        super(RagelCLexer, self).__init__(CLexer, RagelEmbeddedLexer, **options)

    def analyse_text(text):
        return '@LANG: c' in text