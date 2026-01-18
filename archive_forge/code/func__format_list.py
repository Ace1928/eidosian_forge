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
def _format_list(self, extracted_list):
    """Format a list of traceback entry tuples for printing.

        Given a list of tuples as returned by extract_tb() or
        extract_stack(), return a list of strings ready for printing.
        Each string in the resulting list corresponds to the item with the
        same index in the argument list.  Each string ends in a newline;
        the strings may contain internal newlines as well, for those items
        whose source text line is not None.

        Lifted almost verbatim from traceback.py
        """
    Colors = self.Colors
    output_list = []
    for ind, (filename, lineno, name, line) in enumerate(extracted_list):
        normalCol, nameCol, fileCol, lineCol = (Colors.normalEm, Colors.nameEm, Colors.filenameEm, Colors.line) if ind == len(extracted_list) - 1 else (Colors.Normal, Colors.name, Colors.filename, '')
        fns = _format_filename(filename, fileCol, normalCol, lineno=lineno)
        item = f'{normalCol}  {fns}'
        if name != '<module>':
            item += f' in {nameCol}{name}{normalCol}\n'
        else:
            item += '\n'
        if line:
            item += f'{lineCol}    {line.strip()}{normalCol}\n'
        output_list.append(item)
    return output_list