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
def InstallExceptionHandler(handler):
    """Install an exception handler.

  Args:
    handler: an object conforming to the interface defined in ExceptionHandler

  Raises:
    TypeError: handler was not of the correct type

  All installed exception handlers will be called if main() exits via
  an abnormal exception, i.e. not one of SystemExit, KeyboardInterrupt,
  FlagsError or UsageError.
  """
    if not isinstance(handler, ExceptionHandler):
        raise TypeError('handler of type %s does not inherit from ExceptionHandler' % type(handler))
    EXCEPTION_HANDLERS.append(handler)