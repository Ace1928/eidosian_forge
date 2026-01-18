import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
def _tr_magic(line_info: LineInfo):
    """Translate lines escaped with: %"""
    tpl = '%sget_ipython().run_line_magic(%r, %r)'
    if line_info.line.startswith(ESC_MAGIC2):
        return line_info.line
    cmd = ' '.join([line_info.ifun, line_info.the_rest]).strip()
    t_magic_name, _, t_magic_arg_s = cmd.partition(' ')
    t_magic_name = t_magic_name.lstrip(ESC_MAGIC)
    return tpl % (line_info.pre, t_magic_name, t_magic_arg_s)