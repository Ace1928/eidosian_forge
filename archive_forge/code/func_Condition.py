import os
import sys
import threading
from . import process
from . import reduction
def Condition(self, lock=None):
    """Returns a condition object"""
    from .synchronize import Condition
    return Condition(lock, ctx=self.get_context())