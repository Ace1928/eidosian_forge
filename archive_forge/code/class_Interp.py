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
class Interp(ReplInterpreter):

    def __init__(self, locals: Optional[Dict[str, Any]]=None) -> None:
        """Constructor.

        We include an argument for the outfile to pass to the formatter for it
        to write to.
        """
        super().__init__(locals)

        def write(err_line: Union[str, FmtStr]) -> None:
            """Default stderr handler for tracebacks

            Accepts FmtStrs so interpreters can output them"""
            sys.stderr.write(str(err_line))
        self.write = write
        self.outfile = self

    def writetb(self, lines: Iterable[str]) -> None:
        tbtext = ''.join(lines)
        lexer = get_lexer_by_name('pytb')
        self.format(tbtext, lexer)

    def format(self, tbtext: str, lexer: Any) -> None:
        traceback_informative_formatter = BPythonFormatter(default_colors)
        traceback_code_formatter = BPythonFormatter({Token: 'd'})
        no_format_mode = False
        cur_line = []
        for token, text in lexer.get_tokens(tbtext):
            if text.endswith('\n'):
                cur_line.append((token, text))
                if no_format_mode:
                    traceback_code_formatter.format(cur_line, self.outfile)
                    no_format_mode = False
                else:
                    traceback_informative_formatter.format(cur_line, self.outfile)
                cur_line = []
            elif text == '    ' and len(cur_line) == 0:
                no_format_mode = True
                cur_line.append((token, text))
            else:
                cur_line.append((token, text))
        assert cur_line == [], cur_line