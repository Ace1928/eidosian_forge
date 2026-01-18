import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def _obsolete(self, fileno):
    """ We've received an indication that 'fileno' has been obsoleted.
            Any current listeners must be defanged, and notifications to
            their greenlets queued up to send.
        """
    found = False
    for evtype, bucket in self.secondaries.items():
        if fileno in bucket:
            for listener in bucket[fileno]:
                found = True
                self.closed.append(listener)
                listener.defang()
            del bucket[fileno]
    for evtype, bucket in self.listeners.items():
        if fileno in bucket:
            listener = bucket[fileno]
            found = True
            self.closed.append(listener)
            self.remove(listener)
            listener.defang()
    return found