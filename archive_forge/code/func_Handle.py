import errno
import os
import pdb
import socket
import stat
import struct
import sys
import time
import traceback
import gflags as flags
def Handle(self, exc):
    """Do something with the current exception.

    Args:
      exc: Exception, the current exception

    This method must be overridden.
    """
    raise NotImplementedError()