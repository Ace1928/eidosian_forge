import os
import sys
import threading
from . import process
from . import reduction
def Queue(self, maxsize=0):
    """Returns a queue object"""
    from .queues import Queue
    return Queue(maxsize, ctx=self.get_context())