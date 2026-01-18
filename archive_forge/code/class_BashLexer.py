import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches
class BashLexer(RegexLexer):
    """
    Lexer for (ba|k|z|)sh shell scripts.

    .. versionadded:: 0.6
    """
    name = 'Bash'
    aliases = ['bash', 'sh', 'ksh', 'zsh', 'shell']
    filenames = ['*.sh', '*.ksh', '*.bash', '*.ebuild', '*.eclass', '*.exheres-0', '*.exlib', '*.zsh', '.bashrc', 'bashrc', '.bash_*', 'bash_*', 'zshrc', '.zshrc', 'PKGBUILD']
    mimetypes = ['application/x-sh', 'application/x-shellscript']
    tokens = {'root': [include('basic'), ('`', String.Backtick, 'backticks'), include('data'), include('interp')], 'interp': [('\\$\\(\\(', Keyword, 'math'), ('\\$\\(', Keyword, 'paren'), ('\\$\\{#?', String.Interpol, 'curly'), ('\\$[a-zA-Z_]\\w*', Name.Variable), ('\\$(?:\\d+|[#$?!_*@-])', Name.Variable), ('\\$', Text)], 'basic': [('\\b(if|fi|else|while|do|done|for|then|return|function|case|select|continue|until|esac|elif)(\\s*)\\b', bygroups(Keyword, Text)), ('\\b(alias|bg|bind|break|builtin|caller|cd|command|compgen|complete|declare|dirs|disown|echo|enable|eval|exec|exit|export|false|fc|fg|getopts|hash|help|history|jobs|kill|let|local|logout|popd|printf|pushd|pwd|read|readonly|set|shift|shopt|source|suspend|test|time|times|trap|true|type|typeset|ulimit|umask|unalias|unset|wait)(?=[\\s)`])', Name.Builtin), ('\\A#!.+\\n', Comment.Hashbang), ('#.*\\n', Comment.Single), ('\\\\[\\w\\W]', String.Escape), ('(\\b\\w+)(\\s*)(\\+?=)', bygroups(Name.Variable, Text, Operator)), ('[\\[\\]{}()=]', Operator), ('<<<', Operator), ("<<-?\\s*(\\'?)\\\\?(\\w+)[\\w\\W]+?\\2", String), ('&&|\\|\\|', Operator)], 'data': [('(?s)\\$?"(\\\\\\\\|\\\\[0-7]+|\\\\.|[^"\\\\$])*"', String.Double), ('"', String.Double, 'string'), ("(?s)\\$'(\\\\\\\\|\\\\[0-7]+|\\\\.|[^'\\\\])*'", String.Single), ("(?s)'.*?'", String.Single), (';', Punctuation), ('&', Punctuation), ('\\|', Punctuation), ('\\s+', Text), ('\\d+\\b', Number), ('[^=\\s\\[\\]{}()$"\\\'`\\\\<&|;]+', Text), ('<', Text)], 'string': [('"', String.Double, '#pop'), ('(?s)(\\\\\\\\|\\\\[0-7]+|\\\\.|[^"\\\\$])+', String.Double), include('interp')], 'curly': [('\\}', String.Interpol, '#pop'), (':-', Keyword), ('\\w+', Name.Variable), ('[^}:"\\\'`$\\\\]+', Punctuation), (':', Punctuation), include('root')], 'paren': [('\\)', Keyword, '#pop'), include('root')], 'math': [('\\)\\)', Keyword, '#pop'), ('[-+*/%^|&]|\\*\\*|\\|\\|', Operator), ('\\d+#\\d+', Number), ('\\d+#(?! )', Number), ('\\d+', Number), include('root')], 'backticks': [('`', String.Backtick, '#pop'), include('root')]}

    def analyse_text(text):
        if shebang_matches(text, '(ba|z|)sh'):
            return 1
        if text.startswith('$ '):
            return 0.2