import re
from pygments.lexer import RegexLexer, DelegatingLexer, bygroups, include, \
from pygments.token import Punctuation, \
from pygments.util import get_choice_opt, iteritems
from pygments import unistring as uni
from pygments.lexers.html import XmlLexer

    For the F# language (version 3.0).

    AAAAACK Strings
    http://research.microsoft.com/en-us/um/cambridge/projects/fsharp/manual/spec.html#_Toc335818775

    .. versionadded:: 1.5
    