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
class AtomicDict(object):
    """Thread-safe (and optionally process-safe) dictionary protected by a lock.

  If a multiprocessing.Manager is supplied on init, the dictionary is
  both process and thread safe. Otherwise, it is only thread-safe.
  """

    def __init__(self, manager=None):
        """Initializes the dict.

    Args:
      manager: (multiprocessing.Manager or None) Manager instance (required for
          cross-process safety), or none if cross-process safety is not needed.
    """
        if manager:
            self.lock = manager.Lock()
            self.dict = manager.dict()
        else:
            self.lock = threading.Lock()
            self.dict = {}

    def __getitem__(self, key):
        with self.lock:
            return self.dict[key]

    def __setitem__(self, key, value):
        with self.lock:
            self.dict[key] = value

    def get(self, key, default_value=None):
        with self.lock:
            return self.dict.get(key, default_value)

    def delete(self, key):
        with self.lock:
            del self.dict[key]

    def values(self):
        with self.lock:
            return self.dict.values()

    def Increment(self, key, inc, default_value=0):
        """Atomically updates the stored value associated with the given key.

    Performs the atomic equivalent of
    dict[key] = dict.get(key, default_value) + inc.

    Args:
      key: lookup key for the value of the first operand of the "+" operation.
      inc: Second operand of the "+" operation.
      default_value: Default value if there is no existing value for the key.

    Returns:
      Incremented value.
    """
        with self.lock:
            val = self.dict.get(key, default_value) + inc
            self.dict[key] = val
            return val