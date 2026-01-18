import re
import copy
from pygments.lexer import ExtendedRegexLexer, RegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import iteritems
class LessCssLexer(CssLexer):
    """
    For `LESS <http://lesscss.org/>`_ styleshets.

    .. versionadded:: 2.1
    """
    name = 'LessCss'
    aliases = ['less']
    filenames = ['*.less']
    mimetypes = ['text/x-less-css']
    tokens = {'root': [('@\\w+', Name.Variable), inherit], 'content': [('\\{', Punctuation, '#push'), inherit]}