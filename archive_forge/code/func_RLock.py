import os
import sys
import threading
from . import process
from . import reduction
def RLock(self):
    """Returns a recursive lock object"""
    from .synchronize import RLock
    return RLock(ctx=self.get_context())