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
class RagelEmbeddedLexer(RegexLexer):
    """
    A lexer for `Ragel`_ embedded in a host language file.

    This will only highlight Ragel statements. If you want host language
    highlighting then call the language-specific Ragel lexer.

    .. versionadded:: 1.1
    """
    name = 'Embedded Ragel'
    aliases = ['ragel-em']
    filenames = ['*.rl']
    tokens = {'root': [('(' + '|'.join(('[^%\\\'"/#]+', '%(?=[^%]|$)', '"(\\\\\\\\|\\\\"|[^"])*"', "'(\\\\\\\\|\\\\'|[^'])*'", '/\\*(.|\\n)*?\\*/', '//.*$\\n?', '\\#.*$\\n?', '/(?!\\*)(\\\\\\\\|\\\\/|[^/])*/', '/')) + ')+', Other), ('(%%)(?![{%])(.*)($|;)(\\n?)', bygroups(Punctuation, using(RagelLexer), Punctuation, Text)), ('(%%%%|%%)\\{', Punctuation, 'multi-line-fsm')], 'multi-line-fsm': [('(' + '|'.join(('(' + '|'.join(('[^}\\\'"\\[/#]', '\\}(?=[^%]|$)', '\\}%(?=[^%]|$)', '[^\\\\]\\\\[{}]', '(>|\\$|%|<|@|<>)/', '/(?!\\*)(\\\\\\\\|\\\\/|[^/])*/\\*', '/(?=[^/*]|$)')) + ')+', '"(\\\\\\\\|\\\\"|[^"])*"', "'(\\\\\\\\|\\\\'|[^'])*'", '\\[(\\\\\\\\|\\\\\\]|[^\\]])*\\]', '/\\*(.|\\n)*?\\*/', '//.*$\\n?', '\\#.*$\\n?')) + ')+', using(RagelLexer)), ('\\}%%', Punctuation, '#pop')]}

    def analyse_text(text):
        return '@LANG: indep' in text