import sys
from codeop import CommandCompiler
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from pygments.token import Generic, Token, Keyword, Name, Comment, String
from pygments.token import Error, Literal, Number, Operator, Punctuation
from pygments.token import Whitespace, _TokenType
from pygments.formatter import Formatter
from pygments.lexers import get_lexer_by_name
from curtsies.formatstring import FmtStr
from ..curtsiesfrontend.parse import parse
from ..repl import Interpreter as ReplInterpreter
class BPythonFormatter(Formatter):
    """This is subclassed from the custom formatter for bpython.  Its format()
    method receives the tokensource and outfile params passed to it from the
    Pygments highlight() method and slops them into the appropriate format
    string as defined above, then writes to the outfile object the final
    formatted string. This does not write real strings. It writes format string
    (FmtStr) objects.

    See the Pygments source for more info; it's pretty
    straightforward."""

    def __init__(self, color_scheme: Dict[_TokenType, str], **options: Union[str, bool, None]) -> None:
        self.f_strings = {k: f'\x01{v}' for k, v in color_scheme.items()}
        super().__init__(**options)

    def format(self, tokensource, outfile):
        o = ''
        for token, text in tokensource:
            while token not in self.f_strings:
                token = token.parent
            o += f'{self.f_strings[token]}\x03{text}\x04'
        outfile.write(parse(o.rstrip()))