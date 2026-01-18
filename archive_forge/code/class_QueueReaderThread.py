import queue
import re
import subprocess
import sys
import threading
import time
from io import DEFAULT_BUFFER_SIZE
from .exceptions import DecodeError
from .base import AudioFile
class QueueReaderThread(threading.Thread):
    """A thread that consumes data from a filehandle and sends the data
    over a Queue.
    """

    def __init__(self, fh, blocksize=1024, discard=False):
        super().__init__()
        self.fh = fh
        self.blocksize = blocksize
        self.daemon = True
        self.discard = discard
        self.queue = None if discard else queue.Queue()

    def run(self):
        while True:
            data = self.fh.read(self.blocksize)
            if not self.discard:
                self.queue.put(data)
            if not data:
                break