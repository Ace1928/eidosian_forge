import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches
class TcshLexer(RegexLexer):
    """
    Lexer for tcsh scripts.

    .. versionadded:: 0.10
    """
    name = 'Tcsh'
    aliases = ['tcsh', 'csh']
    filenames = ['*.tcsh', '*.csh']
    mimetypes = ['application/x-csh']
    tokens = {'root': [include('basic'), ('\\$\\(', Keyword, 'paren'), ('\\$\\{#?', Keyword, 'curly'), ('`', String.Backtick, 'backticks'), include('data')], 'basic': [('\\b(if|endif|else|while|then|foreach|case|default|continue|goto|breaksw|end|switch|endsw)\\s*\\b', Keyword), ('\\b(alias|alloc|bg|bindkey|break|builtins|bye|caller|cd|chdir|complete|dirs|echo|echotc|eval|exec|exit|fg|filetest|getxvers|glob|getspath|hashstat|history|hup|inlib|jobs|kill|limit|log|login|logout|ls-F|migrate|newgrp|nice|nohup|notify|onintr|popd|printenv|pushd|rehash|repeat|rootnode|popd|pushd|set|shift|sched|setenv|setpath|settc|setty|setxvers|shift|source|stop|suspend|source|suspend|telltc|time|umask|unalias|uncomplete|unhash|universe|unlimit|unset|unsetenv|ver|wait|warp|watchlog|where|which)\\s*\\b', Name.Builtin), ('#.*', Comment), ('\\\\[\\w\\W]', String.Escape), ('(\\b\\w+)(\\s*)(=)', bygroups(Name.Variable, Text, Operator)), ('[\\[\\]{}()=]+', Operator), ("<<\\s*(\\'?)\\\\?(\\w+)[\\w\\W]+?\\2", String), (';', Punctuation)], 'data': [('(?s)"(\\\\\\\\|\\\\[0-7]+|\\\\.|[^"\\\\])*"', String.Double), ("(?s)'(\\\\\\\\|\\\\[0-7]+|\\\\.|[^'\\\\])*'", String.Single), ('\\s+', Text), ('[^=\\s\\[\\]{}()$"\\\'`\\\\;#]+', Text), ('\\d+(?= |\\Z)', Number), ('\\$#?(\\w+|.)', Name.Variable)], 'curly': [('\\}', Keyword, '#pop'), (':-', Keyword), ('\\w+', Name.Variable), ('[^}:"\\\'`$]+', Punctuation), (':', Punctuation), include('root')], 'paren': [('\\)', Keyword, '#pop'), include('root')], 'backticks': [('`', String.Backtick, '#pop'), include('root')]}