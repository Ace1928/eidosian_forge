import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import unirange
from pygments.lexers.css import _indentation, _starts_block
from pygments.lexers.html import HtmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.ruby import RubyLexer
class CirruLexer(RegexLexer):
    """
    Syntax rules of Cirru can be found at:
    http://cirru.org/

    * using ``()`` for expressions, but restricted in a same line
    * using ``""`` for strings, with ``\\`` for escaping chars
    * using ``$`` as folding operator
    * using ``,`` as unfolding operator
    * using indentations for nested blocks

    .. versionadded:: 2.0
    """
    name = 'Cirru'
    aliases = ['cirru']
    filenames = ['*.cirru']
    mimetypes = ['text/x-cirru']
    flags = re.MULTILINE
    tokens = {'string': [('[^"\\\\\\n]', String), ('\\\\', String.Escape, 'escape'), ('"', String, '#pop')], 'escape': [('.', String.Escape, '#pop')], 'function': [('\\,', Operator, '#pop'), ('[^\\s"()]+', Name.Function, '#pop'), ('\\)', Operator, '#pop'), ('(?=\\n)', Text, '#pop'), ('\\(', Operator, '#push'), ('"', String, ('#pop', 'string')), ('[ ]+', Text.Whitespace)], 'line': [('(?<!\\w)\\$(?!\\w)', Operator, 'function'), ('\\(', Operator, 'function'), ('\\)', Operator), ('\\n', Text, '#pop'), ('"', String, 'string'), ('[ ]+', Text.Whitespace), ('[+-]?[\\d.]+\\b', Number), ('[^\\s"()]+', Name.Variable)], 'root': [('^\\n+', Text.Whitespace), default(('line', 'function'))]}