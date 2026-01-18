import re
import copy
from pygments.lexer import ExtendedRegexLexer, RegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import iteritems

    For `LESS <http://lesscss.org/>`_ styleshets.

    .. versionadded:: 2.1
    