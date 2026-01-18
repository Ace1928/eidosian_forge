import re
from pygments.lexers import guess_lexer, get_lexer_by_name
from pygments.lexer import RegexLexer, bygroups, default, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import ClassNotFound
class KernelLogLexer(RegexLexer):
    """
    For Linux Kernel log ("dmesg") output.

    .. versionadded:: 2.6
    """
    name = 'Kernel log'
    aliases = ['kmsg', 'dmesg']
    filenames = ['*.kmsg', '*.dmesg']
    tokens = {'root': [('^[^:]+:debug : (?=\\[)', Text, 'debug'), ('^[^:]+:info  : (?=\\[)', Text, 'info'), ('^[^:]+:warn  : (?=\\[)', Text, 'warn'), ('^[^:]+:notice: (?=\\[)', Text, 'warn'), ('^[^:]+:err   : (?=\\[)', Text, 'error'), ('^[^:]+:crit  : (?=\\[)', Text, 'error'), ('^(?=\\[)', Text, 'unknown')], 'unknown': [('^(?=.+(warning|notice|audit|deprecated))', Text, 'warn'), ('^(?=.+(error|critical|fail|Bug))', Text, 'error'), default('info')], 'base': [('\\[[0-9. ]+\\] ', Number), ('(?<=\\] ).+?:', Keyword), ('\\n', Text, '#pop')], 'debug': [include('base'), ('.+\\n', Comment, '#pop')], 'info': [include('base'), ('.+\\n', Text, '#pop')], 'warn': [include('base'), ('.+\\n', Generic.Strong, '#pop')], 'error': [include('base'), ('.+\\n', Generic.Error, '#pop')]}