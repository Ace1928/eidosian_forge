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
class TBTools(colorable.Colorable):
    """Basic tools used by all traceback printer classes."""
    tb_offset = 0

    def __init__(self, color_scheme='NoColor', call_pdb=False, ostream=None, parent=None, config=None, *, debugger_cls=None):
        super(TBTools, self).__init__(parent=parent, config=config)
        self.call_pdb = call_pdb
        self._ostream = ostream
        self.color_scheme_table = exception_colors()
        self.set_colors(color_scheme)
        self.old_scheme = color_scheme
        self.debugger_cls = debugger_cls or debugger.Pdb
        if call_pdb:
            self.pdb = self.debugger_cls()
        else:
            self.pdb = None

    def _get_ostream(self):
        """Output stream that exceptions are written to.

        Valid values are:

        - None: the default, which means that IPython will dynamically resolve
          to sys.stdout.  This ensures compatibility with most tools, including
          Windows (where plain stdout doesn't recognize ANSI escapes).

        - Any object with 'write' and 'flush' attributes.
        """
        return sys.stdout if self._ostream is None else self._ostream

    def _set_ostream(self, val):
        assert val is None or (hasattr(val, 'write') and hasattr(val, 'flush'))
        self._ostream = val
    ostream = property(_get_ostream, _set_ostream)

    @staticmethod
    def _get_chained_exception(exception_value):
        cause = getattr(exception_value, '__cause__', None)
        if cause:
            return cause
        if getattr(exception_value, '__suppress_context__', False):
            return None
        return getattr(exception_value, '__context__', None)

    def get_parts_of_chained_exception(self, evalue) -> Optional[Tuple[type, BaseException, TracebackType]]:
        chained_evalue = self._get_chained_exception(evalue)
        if chained_evalue:
            return (chained_evalue.__class__, chained_evalue, chained_evalue.__traceback__)
        return None

    def prepare_chained_exception_message(self, cause) -> List[Any]:
        direct_cause = '\nThe above exception was the direct cause of the following exception:\n'
        exception_during_handling = '\nDuring handling of the above exception, another exception occurred:\n'
        if cause:
            message = [[direct_cause]]
        else:
            message = [[exception_during_handling]]
        return message

    @property
    def has_colors(self) -> bool:
        return self.color_scheme_table.active_scheme_name.lower() != 'nocolor'

    def set_colors(self, *args, **kw):
        """Shorthand access to the color table scheme selector method."""
        self.color_scheme_table.set_active_scheme(*args, **kw)
        self.Colors = self.color_scheme_table.active_colors
        if hasattr(self, 'pdb') and self.pdb is not None:
            self.pdb.set_colors(*args, **kw)

    def color_toggle(self):
        """Toggle between the currently active color scheme and NoColor."""
        if self.color_scheme_table.active_scheme_name == 'NoColor':
            self.color_scheme_table.set_active_scheme(self.old_scheme)
            self.Colors = self.color_scheme_table.active_colors
        else:
            self.old_scheme = self.color_scheme_table.active_scheme_name
            self.color_scheme_table.set_active_scheme('NoColor')
            self.Colors = self.color_scheme_table.active_colors

    def stb2text(self, stb):
        """Convert a structured traceback (a list) to a string."""
        return '\n'.join(stb)

    def text(self, etype, value, tb, tb_offset: Optional[int]=None, context=5):
        """Return formatted traceback.

        Subclasses may override this if they add extra arguments.
        """
        tb_list = self.structured_traceback(etype, value, tb, tb_offset, context)
        return self.stb2text(tb_list)

    def structured_traceback(self, etype: type, evalue: Optional[BaseException], etb: Optional[TracebackType]=None, tb_offset: Optional[int]=None, number_of_lines_of_context: int=5):
        """Return a list of traceback frames.

        Must be implemented by each class.
        """
        raise NotImplementedError()