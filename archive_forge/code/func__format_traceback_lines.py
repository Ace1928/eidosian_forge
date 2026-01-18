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
def _format_traceback_lines(lines, Colors, has_colors: bool, lvals):
    """
    Format tracebacks lines with pointing arrow, leading numbers...

    Parameters
    ----------
    lines : list[Line]
    Colors
        ColorScheme used.
    lvals : str
        Values of local variables, already colored, to inject just after the error line.
    """
    numbers_width = INDENT_SIZE - 1
    res = []
    for stack_line in lines:
        if stack_line is stack_data.LINE_GAP:
            res.append('%s   (...)%s\n' % (Colors.linenoEm, Colors.Normal))
            continue
        line = stack_line.render(pygmented=has_colors).rstrip('\n') + '\n'
        lineno = stack_line.lineno
        if stack_line.is_current:
            pad = numbers_width - len(str(lineno))
            num = '%s%s' % (debugger.make_arrow(pad), str(lineno))
            start_color = Colors.linenoEm
        else:
            num = '%*s' % (numbers_width, lineno)
            start_color = Colors.lineno
        line = '%s%s%s %s' % (start_color, num, Colors.Normal, line)
        res.append(line)
        if lvals and stack_line.is_current:
            res.append(lvals + '\n')
    return res