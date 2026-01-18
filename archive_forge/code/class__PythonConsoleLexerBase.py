import re
import keyword
from pygments.lexer import DelegatingLexer, Lexer, RegexLexer, include, \
from pygments.util import get_bool_opt, shebang_matches
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class _PythonConsoleLexerBase(RegexLexer):
    name = 'Python console session'
    aliases = ['pycon']
    mimetypes = ['text/x-python-doctest']
    'Auxiliary lexer for `PythonConsoleLexer`.\n\n    Code tokens are output as ``Token.Other.Code``, traceback tokens as\n    ``Token.Other.Traceback``.\n    '
    tokens = {'root': [('(>>> )(.*\\n)', bygroups(Generic.Prompt, Other.Code), 'continuations'), ('(>>>)(\\n)', bygroups(Generic.Prompt, Whitespace)), ('(\\^C)?Traceback \\(most recent call last\\):\\n', Other.Traceback, 'traceback'), ('  File "[^"]+", line \\d+', Other.Traceback, 'traceback'), ('.*\\n', Generic.Output)], 'continuations': [('(\\.\\.\\. )(.*\\n)', bygroups(Generic.Prompt, Other.Code)), ('(\\.\\.\\.)(\\n)', bygroups(Generic.Prompt, Whitespace)), default('#pop')], 'traceback': [('(?=>>>( |$))', Text, '#pop'), ('(KeyboardInterrupt)(\\n)', bygroups(Name.Class, Whitespace)), ('.*\\n', Other.Traceback)]}