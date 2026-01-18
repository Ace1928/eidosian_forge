from queue import Empty, Queue
from twisted.internet import _threadedselect
from twisted.python import log, runtime
def _installSignalHandlersAgain(self):
    """
        wx sometimes removes our own signal handlers, so re-add them.
        """
    try:
        import signal
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except ImportError:
        return
    self._signals.install()