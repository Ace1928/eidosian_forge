import socket
import selectors
import os
import sys
import threading
from io import BufferedIOBase
from time import monotonic as time
class _NoThreads:
    """
    Degenerate version of _Threads.
    """

    def append(self, thread):
        pass

    def join(self):
        pass