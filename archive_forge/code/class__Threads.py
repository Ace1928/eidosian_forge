import socket
import selectors
import os
import sys
import threading
from io import BufferedIOBase
from time import monotonic as time
class _Threads(list):
    """
    Joinable list of all non-daemon threads.
    """

    def append(self, thread):
        self.reap()
        if thread.daemon:
            return
        super().append(thread)

    def pop_all(self):
        self[:], result = ([], self[:])
        return result

    def join(self):
        for thread in self.pop_all():
            thread.join()

    def reap(self):
        self[:] = (thread for thread in self if thread.is_alive())