import re
from pygments.lexers import (
from pygments.lexer import (
from pygments.token import (
from pygments.util import get_bool_opt
class IPythonPartialTracebackLexer(RegexLexer):
    """
    Partial lexer for IPython tracebacks.

    Handles all the non-python output.

    """
    name = 'IPython Partial Traceback'
    tokens = {'root': [('^(\\^C)?(-+\\n)', bygroups(Error, Generic.Traceback)), ('^(  File)(.*)(, line )(\\d+\\n)', bygroups(Generic.Traceback, Name.Namespace, Generic.Traceback, Literal.Number.Integer)), ('(?u)(^[^\\d\\W]\\w*)(\\s*)(Traceback.*?\\n)', bygroups(Name.Exception, Generic.Whitespace, Text)), ('(.*)( in )(.*)(\\(.*\\)\\n)', bygroups(Name.Namespace, Text, Name.Entity, Name.Tag)), ('(\\s*?)(\\d+)(.*?\\n)', bygroups(Generic.Whitespace, Literal.Number.Integer, Other)), ('(-*>?\\s?)(\\d+)(.*?\\n)', bygroups(Name.Exception, Literal.Number.Integer, Other)), ('(?u)(^[^\\d\\W]\\w*)(:.*?\\n)', bygroups(Name.Exception, Text)), ('.*\\n', Other)]}