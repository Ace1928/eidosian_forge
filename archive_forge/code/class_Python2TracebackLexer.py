import re
import keyword
from pygments.lexer import DelegatingLexer, Lexer, RegexLexer, include, \
from pygments.util import get_bool_opt, shebang_matches
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class Python2TracebackLexer(RegexLexer):
    """
    For Python tracebacks.

    .. versionadded:: 0.7

    .. versionchanged:: 2.5
       This class has been renamed from ``PythonTracebackLexer``.
       ``PythonTracebackLexer`` now refers to the Python 3 variant.
    """
    name = 'Python 2.x Traceback'
    aliases = ['py2tb']
    filenames = ['*.py2tb']
    mimetypes = ['text/x-python2-traceback']
    tokens = {'root': [('^(\\^C)?(Traceback.*\\n)', bygroups(Text, Generic.Traceback), 'intb'), ('^(?=  File "[^"]+", line \\d+)', Generic.Traceback, 'intb'), ('^.*\\n', Other)], 'intb': [('^(  File )("[^"]+")(, line )(\\d+)(, in )(.+)(\\n)', bygroups(Text, Name.Builtin, Text, Number, Text, Name, Whitespace)), ('^(  File )("[^"]+")(, line )(\\d+)(\\n)', bygroups(Text, Name.Builtin, Text, Number, Whitespace)), ('^(    )(.+)(\\n)', bygroups(Text, using(Python2Lexer), Whitespace), 'marker'), ('^([ \\t]*)(\\.\\.\\.)(\\n)', bygroups(Text, Comment, Whitespace)), ('^([^:]+)(: )(.+)(\\n)', bygroups(Generic.Error, Text, Name, Whitespace), '#pop'), ('^([a-zA-Z_]\\w*)(:?\\n)', bygroups(Generic.Error, Whitespace), '#pop')], 'marker': [('( {4,})(\\^)', bygroups(Text, Punctuation.Marker), '#pop'), default('#pop')]}