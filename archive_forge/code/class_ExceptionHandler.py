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
class ExceptionHandler(object):
    """Base exception handler from which other may inherit."""

    def Wants(self, unused_exc):
        """Check if this exception handler want to handle this exception.

    Args:
      unused_exc: Exception, the current exception

    Returns:
      boolean

    This base handler wants to handle all exceptions, override this
    method if you want to be more selective.
    """
        return True

    def Handle(self, exc):
        """Do something with the current exception.

    Args:
      exc: Exception, the current exception

    This method must be overridden.
    """
        raise NotImplementedError()