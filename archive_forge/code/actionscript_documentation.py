import re
from pygments.lexer import RegexLexer, bygroups, using, this, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \

    For MXML markup.
    Nested AS3 in <script> tags is highlighted by the appropriate lexer.

    .. versionadded:: 1.1
    