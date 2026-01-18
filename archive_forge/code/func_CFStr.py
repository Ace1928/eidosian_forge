from six.moves import queue
from six.moves import range
import ctypes
import ctypes.util
import logging
import sys
import threading
from pyu2f import errors
from pyu2f.hid import base
def CFStr(s):
    """Builds a CFString from a python string.

  Args:
    s: source string

  Returns:
    CFStringRef representation of the source string

  Resulting CFString must be CFReleased when no longer needed.
  """
    return cf.CFStringCreateWithCString(None, s.encode(), 0)