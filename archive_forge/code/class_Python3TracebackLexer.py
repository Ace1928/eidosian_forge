import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.util import get_bool_opt, shebang_matches
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class Python3TracebackLexer(RegexLexer):
    """
    For Python 3.0 tracebacks, with support for chained exceptions.

    .. versionadded:: 1.0
    """
    name = 'Python 3.0 Traceback'
    aliases = ['py3tb']
    filenames = ['*.py3tb']
    mimetypes = ['text/x-python3-traceback']
    tokens = {'root': [('\\n', Text), ('^Traceback \\(most recent call last\\):\\n', Generic.Traceback, 'intb'), ('^During handling of the above exception, another exception occurred:\\n\\n', Generic.Traceback), ('^The above exception was the direct cause of the following exception:\\n\\n', Generic.Traceback), ('^(?=  File "[^"]+", line \\d+)', Generic.Traceback, 'intb')], 'intb': [('^(  File )("[^"]+")(, line )(\\d+)(, in )(.+)(\\n)', bygroups(Text, Name.Builtin, Text, Number, Text, Name, Text)), ('^(  File )("[^"]+")(, line )(\\d+)(\\n)', bygroups(Text, Name.Builtin, Text, Number, Text)), ('^(    )(.+)(\\n)', bygroups(Text, using(Python3Lexer), Text)), ('^([ \\t]*)(\\.\\.\\.)(\\n)', bygroups(Text, Comment, Text)), ('^([^:]+)(: )(.+)(\\n)', bygroups(Generic.Error, Text, Name, Text), '#pop'), ('^([a-zA-Z_]\\w*)(:?\\n)', bygroups(Generic.Error, Text), '#pop')]}