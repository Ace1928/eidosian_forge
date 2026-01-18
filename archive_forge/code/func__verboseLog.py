import time
from twisted.internet import protocol
from twisted.names import dns, resolve
from twisted.python import log
def _verboseLog(self, *args, **kwargs):
    """
        Log a message only if verbose logging is enabled.

        @param args: Positional arguments which will be passed to C{log.msg}
        @param kwargs: Keyword arguments which will be passed to C{log.msg}
        """
    if self.verbose > 0:
        log.msg(*args, **kwargs)