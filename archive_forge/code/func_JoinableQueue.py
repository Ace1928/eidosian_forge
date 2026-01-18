import os
import sys
import threading
from . import process
from . import reduction
def JoinableQueue(self, maxsize=0):
    """Returns a queue object"""
    from .queues import JoinableQueue
    return JoinableQueue(maxsize, ctx=self.get_context())