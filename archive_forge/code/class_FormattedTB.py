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
class FormattedTB(VerboseTB, ListTB):
    """Subclass ListTB but allow calling with a traceback.

    It can thus be used as a sys.excepthook for Python > 2.1.

    Also adds 'Context' and 'Verbose' modes, not available in ListTB.

    Allows a tb_offset to be specified. This is useful for situations where
    one needs to remove a number of topmost frames from the traceback (such as
    occurs with python programs that themselves execute other python code,
    like Python shells).  """
    mode: str

    def __init__(self, mode='Plain', color_scheme='Linux', call_pdb=False, ostream=None, tb_offset=0, long_header=False, include_vars=False, check_cache=None, debugger_cls=None, parent=None, config=None):
        self.valid_modes = ['Plain', 'Context', 'Verbose', 'Minimal']
        self.verbose_modes = self.valid_modes[1:3]
        VerboseTB.__init__(self, color_scheme=color_scheme, call_pdb=call_pdb, ostream=ostream, tb_offset=tb_offset, long_header=long_header, include_vars=include_vars, check_cache=check_cache, debugger_cls=debugger_cls, parent=parent, config=config)
        self._join_chars = dict(Plain='', Context='\n', Verbose='\n', Minimal='')
        self.set_mode(mode)

    def structured_traceback(self, etype, value, tb, tb_offset=None, number_of_lines_of_context=5):
        tb_offset = self.tb_offset if tb_offset is None else tb_offset
        mode = self.mode
        if mode in self.verbose_modes:
            return VerboseTB.structured_traceback(self, etype, value, tb, tb_offset, number_of_lines_of_context)
        elif mode == 'Minimal':
            return ListTB.get_exception_only(self, etype, value)
        else:
            self.check_cache()
            return ListTB.structured_traceback(self, etype, value, tb, tb_offset, number_of_lines_of_context)

    def stb2text(self, stb):
        """Convert a structured traceback (a list) to a string."""
        return self.tb_join_char.join(stb)

    def set_mode(self, mode: Optional[str]=None):
        """Switch to the desired mode.

        If mode is not specified, cycles through the available modes."""
        if not mode:
            new_idx = (self.valid_modes.index(self.mode) + 1) % len(self.valid_modes)
            self.mode = self.valid_modes[new_idx]
        elif mode not in self.valid_modes:
            raise ValueError('Unrecognized mode in FormattedTB: <' + mode + '>\nValid modes: ' + str(self.valid_modes))
        else:
            assert isinstance(mode, str)
            self.mode = mode
        self.include_vars = self.mode == self.valid_modes[2]
        self.tb_join_char = self._join_chars[self.mode]

    def plain(self):
        self.set_mode(self.valid_modes[0])

    def context(self):
        self.set_mode(self.valid_modes[1])

    def verbose(self):
        self.set_mode(self.valid_modes[2])

    def minimal(self):
        self.set_mode(self.valid_modes[3])