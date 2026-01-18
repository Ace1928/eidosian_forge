import logging
import logging.config
import os
from importlib import import_module
from importlib.util import find_spec
def _add_trace_level():
    """Wrapper to define custom TRACE level for PennyLane logging"""

    def trace(self, message, *args, **kws):
        """Enable a more verbose mode than DEBUG. Used to enable inspection of function definitions in log messages."""
        self._log(TRACE, message, args, **kws)
    logging.addLevelName(TRACE, 'TRACE')
    logging.TRACE = TRACE
    lc = logging.getLoggerClass()
    lc.trace = trace