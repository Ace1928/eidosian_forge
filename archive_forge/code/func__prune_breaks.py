import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def _prune_breaks(self, filename, lineno):
    """Prune breakpoints for filename:lineno.

        A list of breakpoints is maintained in the Bdb instance and in
        the Breakpoint class.  If a breakpoint in the Bdb instance no
        longer exists in the Breakpoint class, then it's removed from the
        Bdb instance.
        """
    if (filename, lineno) not in Breakpoint.bplist:
        self.breaks[filename].remove(lineno)
    if not self.breaks[filename]:
        del self.breaks[filename]