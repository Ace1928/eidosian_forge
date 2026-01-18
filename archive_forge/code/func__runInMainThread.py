from queue import Empty, Queue
from twisted.internet import _threadedselect
from twisted.python import log, runtime
def _runInMainThread(self, f):
    """
        Schedule function to run in main wx/Twisted thread.

        Called by the select() thread.
        """
    if hasattr(self, 'wxapp'):
        wxCallAfter(f)
    else:
        self._postQueue.put(f)