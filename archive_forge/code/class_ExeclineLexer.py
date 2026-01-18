import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, Whitespace, \
from pygments.util import shebang_matches
class ExeclineLexer(RegexLexer):
    """
    Lexer for Laurent Bercot's execline language
    (https://skarnet.org/software/execline).

    .. versionadded:: 2.7
    """
    name = 'execline'
    aliases = ['execline']
    filenames = ['*.exec']
    tokens = {'root': [include('basic'), include('data'), include('interp')], 'interp': [('\\$\\{', String.Interpol, 'curly'), ('\\$[\\w@#]+', Name.Variable), ('\\$', Text)], 'basic': [('\\b(background|backtick|cd|define|dollarat|elgetopt|elgetpositionals|elglob|emptyenv|envfile|exec|execlineb|exit|export|fdblock|fdclose|fdmove|fdreserve|fdswap|forbacktickx|foreground|forstdin|forx|getcwd|getpid|heredoc|homeof|if|ifelse|ifte|ifthenelse|importas|loopwhilex|multidefine|multisubstitute|pipeline|piperw|posix-cd|redirfd|runblock|shift|trap|tryexec|umask|unexport|wait|withstdinas)\\b', Name.Builtin), ('\\A#!.+\\n', Comment.Hashbang), ('#.*\\n', Comment.Single), ('[{}]', Operator)], 'data': [('(?s)"(\\\\.|[^"\\\\$])*"', String.Double), ('"', String.Double, 'string'), ('\\s+', Text), ('[^\\s{}$"\\\\]+', Text)], 'string': [('"', String.Double, '#pop'), ('(?s)(\\\\\\\\|\\\\.|[^"\\\\$])+', String.Double), include('interp')], 'curly': [('\\}', String.Interpol, '#pop'), ('[\\w#@]+', Name.Variable), include('root')]}

    def analyse_text(text):
        if shebang_matches(text, 'execlineb'):
            return 1