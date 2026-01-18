import os
import sys
import threading
from . import process
from . import reduction
def Semaphore(self, value=1):
    """Returns a semaphore object"""
    from .synchronize import Semaphore
    return Semaphore(value, ctx=self.get_context())