import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
def _line_tokens(line):
    """Helper for has_comment and ends_in_comment_or_string."""
    readline = StringIO(line).readline
    toktypes = set()
    try:
        for t in tokenutil.generate_tokens_catch_errors(readline):
            toktypes.add(t[0])
    except TokenError as e:
        if 'multi-line string' in e.args[0]:
            toktypes.add(_MULTILINE_STRING)
        else:
            toktypes.add(_MULTILINE_STRUCTURE)
    return toktypes