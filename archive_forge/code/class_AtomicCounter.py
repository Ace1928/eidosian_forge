import os
import queue
import socket
import threading
import time
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.record_writer import RecordWriter
class AtomicCounter:

    def __init__(self, initial_value):
        self._value = initial_value
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            try:
                return self._value
            finally:
                self._value += 1