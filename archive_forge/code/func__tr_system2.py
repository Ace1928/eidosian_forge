import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
def _tr_system2(line_info: LineInfo):
    """Translate lines escaped with: !!"""
    cmd = line_info.line.lstrip()[2:]
    return '%sget_ipython().getoutput(%r)' % (line_info.pre, cmd)