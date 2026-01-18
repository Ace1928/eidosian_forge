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
def _elide_point(string: str, *, min_elide=30) -> str:
    """
    If a string is long enough, and has at least 3 dots,
    replace the middle part with ellipses.

    If a string naming a file is long enough, and has at least 3 slashes,
    replace the middle part with ellipses.

    If three consecutive dots, or two consecutive dots are encountered these are
    replaced by the equivalents HORIZONTAL ELLIPSIS or TWO DOT LEADER unicode
    equivalents
    """
    string = string.replace('...', '…')
    string = string.replace('..', '‥')
    if len(string) < min_elide:
        return string
    object_parts = string.split('.')
    file_parts = string.split(os.sep)
    if file_parts[-1] == '':
        file_parts.pop()
    if len(object_parts) > 3:
        return '{}.{}…{}.{}'.format(object_parts[0], object_parts[1][:1], object_parts[-2][-1:], object_parts[-1])
    elif len(file_parts) > 3:
        return ('{}' + os.sep + '{}…{}' + os.sep + '{}').format(file_parts[0], file_parts[1][:1], file_parts[-2][-1:], file_parts[-1])
    return string