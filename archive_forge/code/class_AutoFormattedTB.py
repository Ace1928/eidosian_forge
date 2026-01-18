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
class AutoFormattedTB(FormattedTB):
    """A traceback printer which can be called on the fly.

    It will find out about exceptions by itself.

    A brief example::

        AutoTB = AutoFormattedTB(mode = 'Verbose',color_scheme='Linux')
        try:
          ...
        except:
          AutoTB()  # or AutoTB(out=logfile) where logfile is an open file object
    """

    def __call__(self, etype=None, evalue=None, etb=None, out=None, tb_offset=None):
        """Print out a formatted exception traceback.

        Optional arguments:
          - out: an open file-like object to direct output to.

          - tb_offset: the number of frames to skip over in the stack, on a
          per-call basis (this overrides temporarily the instance's tb_offset
          given at initialization time."""
        if out is None:
            out = self.ostream
        out.flush()
        out.write(self.text(etype, evalue, etb, tb_offset))
        out.write('\n')
        out.flush()
        try:
            self.debugger()
        except KeyboardInterrupt:
            print('\nKeyboardInterrupt')

    def structured_traceback(self, etype: type, evalue: Optional[BaseException], etb: Optional[TracebackType]=None, tb_offset: Optional[int]=None, number_of_lines_of_context: int=5):
        if etype is None:
            etype, evalue, etb = sys.exc_info()
        if isinstance(etb, tuple):
            self.tb = etb[0]
        else:
            self.tb = etb
        return FormattedTB.structured_traceback(self, etype, evalue, etb, tb_offset, number_of_lines_of_context)