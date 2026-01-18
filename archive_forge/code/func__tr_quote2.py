import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
def _tr_quote2(line_info: LineInfo):
    """Translate lines escaped with: ;"""
    return '%s%s("%s")' % (line_info.pre, line_info.ifun, line_info.the_rest)