import os
import sys
import threading
from . import process
from . import reduction
def Lock(self):
    """Returns a non-recursive lock object"""
    from .synchronize import Lock
    return Lock(ctx=self.get_context())