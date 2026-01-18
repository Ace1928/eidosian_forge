import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def _load_breaks(self):
    """Apply all breakpoints (set in other instances) to this one.

        Populates this instance's breaks list from the Breakpoint class's
        list, which can have breakpoints set by another Bdb instance. This
        is necessary for interactive sessions to keep the breakpoints
        active across multiple calls to run().
        """
    for filename, lineno in Breakpoint.bplist.keys():
        self._add_to_breaks(filename, lineno)