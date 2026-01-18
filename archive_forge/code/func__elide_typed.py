import unicodedata
from wcwidth import wcwidth
from IPython.core.completer import (
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.patch_stdout import patch_stdout
import pygments.lexers as pygments_lexers
import os
import sys
import traceback
def _elide_typed(string: str, typed: str, *, min_elide: int=30) -> str:
    """
    Elide the middle of a long string if the beginning has already been typed.
    """
    if len(string) < min_elide:
        return string
    cut_how_much = len(typed) - 3
    if cut_how_much < 7:
        return string
    if string.startswith(typed) and len(string) > len(typed):
        return f'{string[:3]}…{string[cut_how_much:]}'
    return string