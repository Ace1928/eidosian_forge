import os
import sys
import threading
from . import process
from . import reduction
def BoundedSemaphore(self, value=1):
    """Returns a bounded semaphore object"""
    from .synchronize import BoundedSemaphore
    return BoundedSemaphore(value, ctx=self.get_context())