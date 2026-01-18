import struct
import socket
import functools
import time
import logging
import Pyro4
class QueueLogger(logging.StreamHandler):

    def __init__(self, queue):
        self._queue = queue
        return super().__init__(None)

    def flush(self):
        pass

    def emit(self, record):
        self._queue.append((self.level, record))