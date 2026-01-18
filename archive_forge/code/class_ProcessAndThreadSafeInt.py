from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import errno
import logging
import multiprocessing
import threading
import traceback
from gslib.utils import constants
from gslib.utils import system_util
from six.moves import queue as Queue
class ProcessAndThreadSafeInt(object):
    """This class implements a process and thread-safe integer.

  It is backed either by a multiprocessing Value of type 'i' or an internal
  threading lock.  This simplifies the calling pattern for
  global variables that could be a Multiprocessing.Value or an integer.
  Without this class, callers need to write code like this:

  global variable_name
  if isinstance(variable_name, int):
    return variable_name
  else:
    return variable_name.value
  """

    def __init__(self, multiprocessing_is_available):
        self.multiprocessing_is_available = multiprocessing_is_available
        if self.multiprocessing_is_available:
            self.value = multiprocessing_context.Value('i', 0)
        else:
            self.lock = threading.Lock()
            self.value = 0

    def Reset(self, reset_value=0):
        if self.multiprocessing_is_available:
            self.value.value = reset_value
        else:
            with self.lock:
                self.value = reset_value

    def Increment(self):
        if self.multiprocessing_is_available:
            self.value.value += 1
        else:
            with self.lock:
                self.value += 1

    def Decrement(self):
        if self.multiprocessing_is_available:
            self.value.value -= 1
        else:
            with self.lock:
                self.value -= 1

    def GetValue(self):
        if self.multiprocessing_is_available:
            return self.value.value
        else:
            with self.lock:
                return self.value