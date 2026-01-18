import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
def _tr_help(line_info: LineInfo):
    """Translate lines escaped with: ?/??"""
    if not line_info.line[1:]:
        return 'get_ipython().show_usage()'
    return _make_help_call(line_info.ifun, line_info.esc, line_info.pre)