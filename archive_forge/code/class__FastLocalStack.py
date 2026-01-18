import os
import socket
import sys
import threading
import traceback
from contextlib import contextmanager
from threading import TIMEOUT_MAX as THREAD_TIMEOUT_MAX
from celery.local import Proxy
class _FastLocalStack(threading.local):

    def __init__(self):
        self.stack = []
        self.push = self.stack.append
        self.pop = self.stack.pop
        super().__init__()

    @property
    def top(self):
        try:
            return self.stack[-1]
        except (AttributeError, IndexError):
            return None

    def __len__(self):
        return len(self.stack)