from collections.abc import Sequence
import functools
import inspect
import linecache
import pydoc
import sys
import time
import traceback
import types
from types import TracebackType
from typing import Any, List, Optional, Tuple
import stack_data
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.styles import get_style_by_name
import IPython.utils.colorable as colorable
from IPython import get_ipython
from IPython.core import debugger
from IPython.core.display_trap import DisplayTrap
from IPython.core.excolors import exception_colors
from IPython.utils import PyColorize
from IPython.utils import path as util_path
from IPython.utils import py3compat
from IPython.utils.terminal import get_terminal_size
def _format_filename(file, ColorFilename, ColorNormal, *, lineno=None):
    """
    Format filename lines with custom formatting from caching compiler or `File *.py` by default

    Parameters
    ----------
    file : str
    ColorFilename
        ColorScheme's filename coloring to be used.
    ColorNormal
        ColorScheme's normal coloring to be used.
    """
    ipinst = get_ipython()
    if ipinst is not None and (data := ipinst.compile.format_code_name(file)) is not None:
        label, name = data
        if lineno is None:
            tpl_link = f'{{label}} {ColorFilename}{{name}}{ColorNormal}'
        else:
            tpl_link = f'{{label}} {ColorFilename}{{name}}, line {{lineno}}{ColorNormal}'
    else:
        label = 'File'
        name = util_path.compress_user(py3compat.cast_unicode(file, util_path.fs_encoding))
        if lineno is None:
            tpl_link = f'{{label}} {ColorFilename}{{name}}{ColorNormal}'
        else:
            tpl_link = f'{{label}} {ColorFilename}{{name}}:{{lineno}}{ColorNormal}'
    return tpl_link.format(label=label, name=name, lineno=lineno)