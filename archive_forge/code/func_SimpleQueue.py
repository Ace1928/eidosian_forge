import os
import sys
import threading
from . import process
from . import reduction
def SimpleQueue(self):
    """Returns a queue object"""
    from .queues import SimpleQueue
    return SimpleQueue(ctx=self.get_context())