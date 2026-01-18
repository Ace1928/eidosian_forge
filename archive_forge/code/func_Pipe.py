import os
import sys
import threading
from . import process
from . import reduction
def Pipe(self, duplex=True):
    """Returns two connection object connected by a pipe"""
    from .connection import Pipe
    return Pipe(duplex)