import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
def _tr_system(line_info: LineInfo):
    """Translate lines escaped with: !"""
    cmd = line_info.line.lstrip().lstrip(ESC_SHELL)
    return '%sget_ipython().system(%r)' % (line_info.pre, cmd)